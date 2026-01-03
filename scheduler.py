import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import pulp

from models import (
    Course,
    Faculty,
    Room,
    TimeSlot,
    TimetableEntry,
    StudentGroup,
    PeriodConfig,
)


@dataclass(frozen=True)
class Session:
    """Represents an atomic lecture hour that must be scheduled."""

    id: int
    course_id: int
    course_code: str
    course_type: str
    student_group: str
    is_lab: bool


class TimetableGenerator:
    """
    Enhanced Hybrid scheduler with 12 advanced constraints:
    
    1. Automatic faculty workload management (min/max bounds)
    2. Priority lab allocation for courses requiring labs
    3. Faculty availability-based prioritization
    4. Per-day faculty timetable generation based on expertise & workload
    5. Prevention of consecutive lectures of same subject
    6. Preferred timeslot assignment for senior faculty
    7. Automatic daily lecture & lab tracking per faculty
    8. Multi-course faculty scheduling with cross-course coordination
    9. Overwork detection & alerts (40+ hours/week warning)
    10. No consecutive same lectures in a day (prevents student boredom)
    11. Avoid same lab on same day (distributes labs across week)
    12. Maximum 2-3 labs per day per group (configurable, prevents overload)
    """

    def __init__(self, db_session, random_seed: int | None = None, config: dict = None, courses=None, groups=None):
        self.db = db_session
        self.random = random.Random(random_seed or random.randint(1, 999_999))
        
        # Store pre-filtered data if provided
        self.courses = courses  # Optional: pre-filtered course list
        self.student_groups = groups  # Optional: pre-filtered student group list
        
        # Enhanced configuration options
        self.config = config or {}
        self.overwork_threshold = self.config.get('overwork_threshold', 40)  # hours/week
        self.senior_faculty_preference = self.config.get('senior_faculty_preference', True)
        self.consecutive_penalty_weight = self.config.get('consecutive_penalty', 20)
        self.lab_priority_weight = self.config.get('lab_priority', 50)
        # Fast mode drastically reduces variable count by ignoring rooms in ILP
        self.fast_mode = self.config.get('fast_mode', True)
        self.verbose = self.config.get('verbose', False)
        # Ultra fast mode: skip ILP completely and use greedy heuristic
        self.ultra_fast = self.config.get('ultra_fast', True)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def generate(self, filters=None):
        import time
        gen_start = time.time()
        context = self._load_context(filters)
        if self.verbose:
            print(f"[GEN] Context: courses={len(context['courses'])}, groups={len(context['student_groups'])}, faculty={len(context['faculty'])}, rooms={len(context['rooms'])}, slots={len(context['time_slots'])}, sessions={len(context['sessions'])}")
        if not context["courses"]:
            return {"success": False, "error": "No courses found matching criteria. Please check filters."}
        if not context["faculty"]:
            return {"success": False, "error": "No faculty found. Please add faculty first."}
        if not context["rooms"]:
            return {"success": False, "error": "No rooms found. Please add rooms first."}
        if not context["time_slots"]:
            return {"success": False, "error": "No time slots found. Please configure time slots."}

        # Ultra fast: greedy assignment pipeline
        if self.ultra_fast:
            load_time = time.time() - gen_start
            greedy_res = self._generate_greedy(context)
            greedy_time = greedy_res.get("greedy_time", 0)
            
            # OPTIMIZATION: Check placement rate - if greedy places >=70% of sessions, accept it
            total_sessions = len(context["sessions"])
            placement_rate = greedy_res.get("placement_rate", 0)
            greedy_threshold = self.config.get('greedy_success_threshold', 0.7)  # 70% default
            
            if greedy_res["success"] and placement_rate >= greedy_threshold:
                # Greedy succeeded with good placement rate - use it!
                warnings = greedy_res.get("warnings", [])
                final_assignments = greedy_res["assignments"]
                
                # Skip overwork detection in ultra-fast mode for speed (optional)
                if not self.config.get('skip_overwork_check', False):
                    overwork_start = time.time()
                    overwork_warnings = self._detect_overwork(final_assignments, context)
                    warnings.extend(overwork_warnings)
                    overwork_time = time.time() - overwork_start
                else:
                    overwork_time = 0
                
                # Skip faculty schedules generation for speed (can be generated on-demand)
                faculty_schedules = {} if self.config.get('skip_faculty_schedules', True) else self._generate_faculty_schedules(final_assignments, context)
                
                persist_start = time.time()
                entries_created = self._persist_assignments(final_assignments, context)
                persist_time = time.time() - persist_start
                
                gen_time = time.time() - gen_start
                
                if self.verbose:
                    print(f"[PERF] Load: {load_time:.2f}s, Greedy: {greedy_time:.2f}s, Overwork: {overwork_time:.2f}s, Persist: {persist_time:.2f}s, Total: {gen_time:.2f}s")
                
                return {
                    "success": True,
                    "entries_created": entries_created,
                    "warnings": warnings,
                    "faculty_schedules": faculty_schedules,
                    "overwork_alerts": [w for w in warnings if "overwork" in w.lower()],
                    "generation_time": gen_time,
                    "performance": {
                        "load_time": load_time,
                        "greedy_time": greedy_time,
                        "overwork_time": overwork_time,
                        "persist_time": persist_time,
                        "total_time": gen_time,
                        "placement_rate": placement_rate,
                        "method": "greedy"
                    }
                }
            elif greedy_res["success"] and placement_rate < greedy_threshold:
                # Greedy succeeded but low placement rate - fallback to ILP
                if self.verbose:
                    print(f"[GEN] Greedy placed {placement_rate*100:.1f}% ({len(greedy_res['assignments'])}/{total_sessions}), falling back to ILP...")
                ilp_start = time.time()
                ilp_fast = self._solve_with_ilp_fast(context)
                ilp_time = time.time() - ilp_start
                
                if not ilp_fast.get('success'):
                    # ILP failed, return greedy result anyway
                    warnings = greedy_res.get("warnings", [])
                    final_assignments = greedy_res["assignments"]
                    persist_start = time.time()
                    entries_created = self._persist_assignments(final_assignments, context)
                    persist_time = time.time() - persist_start
                    gen_time = time.time() - gen_start
                    return {
                        "success": True,
                        "entries_created": entries_created,
                        "warnings": warnings,
                        "faculty_schedules": {},
                        "overwork_alerts": [],
                        "generation_time": gen_time,
                        "performance": {
                            "load_time": load_time,
                            "greedy_time": greedy_time,
                            "ilp_time": ilp_time,
                            "persist_time": persist_time,
                            "total_time": gen_time,
                            "placement_rate": placement_rate,
                            "method": "greedy_fallback"
                        }
                    }
                
                final_assignments = ilp_fast["assignments"]
                warnings = greedy_res.get("warnings", []) + ilp_fast.get("warnings", [])
            else:
                # Greedy failed completely - fallback to ILP
                if self.verbose:
                    print(f"[GEN] Greedy failed: {greedy_res.get('error')}. Falling back to fast ILP…")
                ilp_start = time.time()
                ilp_fast = self._solve_with_ilp_fast(context)
                ilp_time = time.time() - ilp_start
                
                if not ilp_fast.get('success'):
                    return greedy_res
                
                final_assignments = ilp_fast["assignments"]
                warnings = greedy_res.get("warnings", []) + ilp_fast.get("warnings", [])
            
            # Process final assignments (from ILP fallback)
            if not self.config.get('skip_overwork_check', False):
                overwork_start = time.time()
                overwork_warnings = self._detect_overwork(final_assignments, context)
                warnings.extend(overwork_warnings)
                overwork_time = time.time() - overwork_start
            else:
                overwork_time = 0
            
            faculty_schedules = {} if self.config.get('skip_faculty_schedules', True) else self._generate_faculty_schedules(final_assignments, context)
            
            persist_start = time.time()
            entries_created = self._persist_assignments(final_assignments, context)
            persist_time = time.time() - persist_start
            
            gen_time = time.time() - gen_start
            
            if self.verbose:
                print(f"[PERF] Load: {load_time:.2f}s, Greedy: {greedy_time:.2f}s, ILP: {ilp_time:.2f}s, Overwork: {overwork_time:.2f}s, Persist: {persist_time:.2f}s, Total: {gen_time:.2f}s")
            
            return {
                "success": True,
                "entries_created": entries_created,
                "warnings": warnings,
                "faculty_schedules": faculty_schedules,
                "overwork_alerts": [w for w in warnings if "overwork" in w.lower()],
                "generation_time": gen_time,
                "performance": {
                    "load_time": load_time,
                    "greedy_time": greedy_time,
                    "ilp_time": ilp_time if 'ilp_time' in locals() else 0,
                    "overwork_time": overwork_time,
                    "persist_time": persist_time,
                    "total_time": gen_time,
                    "method": "ilp_fallback"
                }
            }

        # Constraint 1: Validate workload bounds (ILP modes)
        bound_report = self._run_bound_analyzer(context)
        if not bound_report["feasible"]:
            return {
                "success": False,
                "error": "Bound analysis failed – please review constraints.",
                "warnings": bound_report["warnings"],
            }

        # Constraint 2 & 3: ILP with lab priority and availability focus
        ilp_result = self._solve_with_ilp(context)
        warnings = bound_report["warnings"] + ilp_result.get("warnings", [])
        if not ilp_result["success"]:
            return {"success": False, "error": ilp_result["error"], "warnings": warnings}

        # Constraints 4-8: Optional GA refinement (skip in fast mode for speed)
        if self.fast_mode:
            final_assignments = ilp_result["assignments"]
        else:
            ga_result = self._refine_with_genetic_algorithm(
                context,
                ilp_result["assignments"],
                ilp_result.get("session_candidates", {}),
            )
            warnings.extend(ga_result.get("warnings", []))
            final_assignments = ga_result.get("assignments", ilp_result["assignments"])

        # Constraint 9: Overwork detection
        overwork_warnings = self._detect_overwork(final_assignments, context)
        warnings.extend(overwork_warnings)

        # Constraint 7: Generate per-faculty daily schedules
        faculty_schedules = self._generate_faculty_schedules(final_assignments, context)
        
        entries_created = self._persist_assignments(final_assignments, context)
        
        return {
            "success": True,
            "entries_created": entries_created,
            "warnings": warnings,
            "faculty_schedules": faculty_schedules,
            "overwork_alerts": [w for w in warnings if "overwork" in w.lower()]
        }

    # --------------------------------------------------------------------- #
    # Ultra-Fast Greedy Scheduler (No ILP)
    # --------------------------------------------------------------------- #
    def _generate_greedy(self, context):
        """Very fast heuristic scheduler that assigns sessions greedily.
        Prioritizes labs first, respects faculty availability, group/day max,
        room capabilities, and avoids same-slot conflicts.
        OPTIMIZED: Uses pre-computed caches, smart slot ordering, early exit.
        CROSS-DEPARTMENT: Respects existing assignments to prevent overlaps.
        """
        import time
        greedy_start = time.time()
        warnings = []
        assignments = []

        # Indexes and helpers
        slot_by_id = context["slot_by_id"]
        time_slots = context["time_slots"]
        course_by_id = context["course_by_id"]
        rooms = context["rooms"]
        faculty_avail = context["faculty_availability"]
        max_per_day = context.get('max_periods_per_day_per_group', 0) or None

        # OPTIMIZATION: Use pre-computed caches (O(1) lookup instead of recomputing)
        course_faculty_cache = context.get("course_faculty_cache", {})
        course_room_cache = context.get("course_room_cache", {})
        
        # Fallback: compute on-demand if cache not available
        if not course_faculty_cache or not course_room_cache:
            expertise_map = context["faculty_expertise"]
            room_caps = context["room_capabilities"]
            course_faculty_cache = {}
            course_room_cache = {}
            for course in context["courses"]:
                course_faculty_cache[course.id] = self._faculty_for_course(course, context["faculty"], expertise_map)
                course_room_cache[course.id] = self._rooms_for_course(course, rooms, room_caps)

        # State trackers
        used_faculty_slot = defaultdict(set)    # (faculty_id) -> {slot_id}
        used_group_slot = defaultdict(set)      # (group_name) -> {slot_id}
        used_room_slot = defaultdict(set)       # (slot_id) -> {room_id}
        group_day_count = defaultdict(lambda: defaultdict(int))  # group -> day -> count
        faculty_hours = defaultdict(int)
        
        # CROSS-DEPARTMENT TRACKING: Initialize with existing assignments
        faculty_existing_slots = context.get("faculty_existing_slots", defaultdict(set))
        faculty_existing_hours = context.get("faculty_existing_hours", defaultdict(int))
        
        # Pre-populate used_faculty_slot with existing assignments to prevent conflicts
        for fac_id, slot_ids in faculty_existing_slots.items():
            used_faculty_slot[fac_id].update(slot_ids)
        
        # Initialize faculty_hours with existing hours
        for fac_id, hours in faculty_existing_hours.items():
            faculty_hours[fac_id] = hours
        
        if self.verbose and faculty_existing_hours:
            print(f"[GREEDY] Initialized with {sum(faculty_existing_hours.values())} existing faculty assignments")
            print(f"[GREEDY] {len(faculty_existing_hours)} faculty members have existing schedules")
        
        # NEW CONSTRAINT TRACKERS
        # Track course placements per group per day: group -> day -> {course_id}
        group_day_courses = defaultdict(lambda: defaultdict(set))
        # Track course placements per group per day per period: group -> day -> period -> course_id
        group_day_period_course = defaultdict(lambda: defaultdict(dict))
        # Track lab count per group per day: group -> day -> lab_count
        group_day_lab_count = defaultdict(lambda: defaultdict(int))

        # Index rooms by type for graceful fallbacks
        rooms_by_type = {
            'lab': [r for r in rooms if getattr(r, 'room_type', '') == 'lab'],
            'classroom': [r for r in rooms if getattr(r, 'room_type', '') == 'classroom']
        }

        # Sort sessions: labs first, then by course code for stability
        sessions = list(context["sessions"]) or []
        sessions.sort(key=lambda s: (0 if s.is_lab else 1, s.course_code))
        
        # OPTIMIZATION: Sequential slot filling to minimize blank periods
        # Fill slots in order (day by day, period by period) to reduce gaps
        slots_sequential = sorted(time_slots, key=lambda s: (s.day, s.period))  # Sequential filling
        
        # Get available days and calculate target fill per day (70% minimum)
        available_days = list(set(slot.day for slot in time_slots))
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        available_days.sort(key=lambda d: days_order.index(d) if d in days_order else 999)
        
        # Calculate periods per day
        periods_per_day_map = defaultdict(int)
        for slot in time_slots:
            periods_per_day_map[slot.day] += 1
        
        total_sessions = len(sessions)
        min_fill_per_day = {day: max(1, int(periods_per_day_map[day] * 0.7)) for day in available_days}  # 70% minimum

        # Iterate each session and place it
        for session in sessions:
            course = course_by_id.get(session.course_id)
            if not course:
                continue

            # OPTIMIZATION: Use pre-computed cache (O(1) lookup)
            cand_faculty = course_faculty_cache.get(course.id, [])
            # Filter by workload
            cand_faculty = [f for f in cand_faculty if faculty_hours[f.id] < (f.max_hours_per_week or 16)]
            if not cand_faculty:
                warnings.append(f"⚠️ No eligible faculty for course {course.code}")
                continue

            # Prefer faculty with lower current hours (load balancing)
            cand_faculty.sort(key=lambda f: faculty_hours[f.id])

            # OPTIMIZATION: Use pre-computed room cache (O(1) lookup)
            eligible_rooms = course_room_cache.get(course.id, [])
            if not eligible_rooms:
                # Graceful fallback: use room type buckets
                fallback = rooms_by_type['lab'] if session.is_lab else rooms_by_type['classroom']
                eligible_rooms = fallback if fallback else rooms
                if not eligible_rooms:
                    warnings.append(f"⚠️ No rooms available to place course {course.code}")
                    continue

            # OPTIMIZATION: 70% fill rate + NO consecutive blank periods (must have 1 filled gap between blanks)
            # Strategy: Distribute evenly across all days, ensure 70% fill, prevent consecutive blanks
            group_occupied_slots = used_group_slot[session.student_group]
            
            # Count how many classes each day has for this group
            group_day_usage = defaultdict(int)
            group_days_used = set()
            group_periods_by_day = defaultdict(list)  # Track which periods are occupied on each day
            
            for occupied_slot_id in group_occupied_slots:
                occupied_slot = slot_by_id.get(occupied_slot_id)
                if occupied_slot:
                    group_day_usage[occupied_slot.day] += 1
                    group_days_used.add(occupied_slot.day)
                    group_periods_by_day[occupied_slot.day].append(occupied_slot.period)
            
            # Sort periods for each day to detect gaps
            for day in group_periods_by_day:
                group_periods_by_day[day].sort()
            
            # Calculate which days are under-filled (below 70% target)
            underfilled_days = set()
            for day in available_days:
                if group_day_usage[day] < min_fill_per_day[day]:
                    underfilled_days.add(day)
            
            # Check if adding this slot would create consecutive blank periods
            def would_create_consecutive_blanks(slot, occupied_periods):
                """Check if NOT placing in this slot would create 2+ consecutive blanks"""
                if not occupied_periods:
                    return False
                
                # Get all periods for this day
                day_periods = [s.period for s in time_slots if s.day == slot.day]
                if not day_periods:
                    return False
                
                min_period = min(day_periods)
                max_period = max(day_periods)
                
                # Check periods around this slot
                # If slot.period-1 and slot.period+1 are both blank, this creates isolated blank
                prev_occupied = (slot.period - 1) in occupied_periods or slot.period - 1 < min_period
                next_occupied = (slot.period + 1) in occupied_periods or slot.period + 1 > max_period
                
                # If neither neighbor would be occupied, check for consecutive blanks
                if not prev_occupied and not next_occupied:
                    # This would create a gap with no filled period between blanks
                    return False
                
                # Check if skipping this slot creates 2+ consecutive blanks between occupied periods
                for i in range(len(occupied_periods) - 1):
                    # Check gap between consecutive occupied periods
                    if occupied_periods[i] < slot.period < occupied_periods[i + 1]:
                        gap_size = occupied_periods[i + 1] - occupied_periods[i] - 1
                        if gap_size >= 2:
                            # There's a gap with 2+ blanks, this slot must be filled
                            return True
                
                return False
            
            # Detect gap slots (blank periods between occupied periods that would violate the rule)
            def is_critical_gap_filler(slot):
                """Check if this slot MUST be filled to prevent consecutive blanks"""
                if slot.day not in group_periods_by_day:
                    return False
                
                occupied_periods = group_periods_by_day[slot.day]
                if len(occupied_periods) < 2:
                    return False
                
                # Check if slot fills a gap that would otherwise have consecutive blanks
                for i in range(len(occupied_periods) - 1):
                    if occupied_periods[i] < slot.period < occupied_periods[i + 1]:
                        gap_size = occupied_periods[i + 1] - occupied_periods[i] - 1
                        # If gap is exactly 2, we MUST fill one to avoid consecutive blanks
                        if gap_size == 2:
                            return True
                        # If gap is 3+, filling this helps but isn't critical yet
                        if gap_size >= 3:
                            return True
                return False
            
            # Smart slot ordering: 70% fill + prevent consecutive blanks
            def slot_priority(slot):
                day_idx = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'].index(slot.day) if slot.day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'] else 0
                
                day_has_classes = slot.day in group_days_used
                classes_on_this_day = group_day_usage.get(slot.day, 0)
                is_underfilled = slot.day in underfilled_days
                is_critical_filler = is_critical_gap_filler(slot)
                
                # Calculate proximity to already scheduled slots on SAME day
                same_day_proximity = 0
                if day_has_classes:
                    occupied_periods = group_periods_by_day[slot.day]
                    
                    # Check if this slot is between two occupied periods (gap filler)
                    is_between = any(occupied_periods[i] < slot.period < occupied_periods[i+1] 
                                   for i in range(len(occupied_periods)-1)) if len(occupied_periods) >= 2 else False
                    
                    # Calculate distance to nearest occupied period
                    min_distance = min([abs(slot.period - p) for p in occupied_periods]) if occupied_periods else 999
                    
                    if min_distance == 1:
                        same_day_proximity = 2000  # Immediately adjacent
                    elif min_distance == 2:
                        same_day_proximity = 1000  # One slot gap
                    elif is_between:
                        same_day_proximity = 800   # Between two classes (fill the gap)
                    elif min_distance <= 3:
                        same_day_proximity = 400   # Close proximity
                    else:
                        same_day_proximity = 100   # Same day
                
                # Priority levels:
                # -2: CRITICAL - Must fill to prevent consecutive blanks
                # -1: URGENT - Underfilled days with gaps that need filling
                #  0: HIGH - Underfilled days with adjacent slots
                #  1: MEDIUM - Empty underfilled days (start new sequence)
                #  2: NORMAL - Days with classes (continue filling)
                #  3: LOW - Sequential filling
                
                if is_critical_filler:
                    return (-2, -same_day_proximity, slot.period)
                elif is_underfilled and day_has_classes and same_day_proximity >= 800:
                    return (-1, -same_day_proximity, -classes_on_this_day, slot.period)
                elif is_underfilled and day_has_classes:
                    return (0, -same_day_proximity, -classes_on_this_day, slot.period)
                elif is_underfilled and not day_has_classes:
                    return (1, day_idx, slot.period)
                elif day_has_classes and same_day_proximity >= 1000:
                    return (2, -same_day_proximity, slot.period)
                else:
                    return (3, day_idx, slot.period)
            
            slot_order = sorted(slots_sequential, key=slot_priority)
            
            placed = False
            # OPTIMIZATION: Early exit - break immediately when placed
            for slot in slot_order:
                day = slot.day
                slot_id = slot.id

                # Group constraints: avoid conflicts and per-day max
                if slot_id in used_group_slot[session.student_group]:
                    continue
                if max_per_day is not None and group_day_count[session.student_group][day] >= max_per_day:
                    continue
                
                # NEW CONSTRAINT 1: No consecutive same lectures in a day
                # Check if placing this course in this period would create consecutive same course
                if slot.period > 1:  # Check previous period
                    prev_course = group_day_period_course[session.student_group][day].get(slot.period - 1)
                    if prev_course == course.id:
                        continue  # Skip: would create consecutive same course
                
                # Also check next period to avoid boxing in
                if slot.period < max(periods_per_day_map.values()):
                    next_course = group_day_period_course[session.student_group][day].get(slot.period + 1)
                    if next_course == course.id:
                        continue  # Skip: would create consecutive same course
                
                # NEW CONSTRAINT 2: Avoid same lab on same day (soft constraint - prefer not to schedule)
                if session.is_lab:
                    # Check if this course already has a lab scheduled on this day
                    if course.id in group_day_courses[session.student_group][day]:
                        # Check if there's a lab already scheduled for this course today
                        already_has_lab_today = any(
                            group_day_period_course[session.student_group][day].get(p) == course.id
                            for p in group_day_period_course[session.student_group][day].keys()
                        )
                        if already_has_lab_today:
                            # Try to find another day first (soft constraint)
                            # We'll only allow if no other day is available (checked later)
                            has_other_day_option = any(
                                other_slot.day != day and 
                                other_slot.id not in used_group_slot[session.student_group]
                                for other_slot in slot_order[:20]  # Check first 20 slots
                            )
                            if has_other_day_option:
                                continue  # Skip this day, prefer another day
                    
                    # NEW CONSTRAINT 3: Maximum 2-3 labs per day
                    current_lab_count = group_day_lab_count[session.student_group][day]
                    max_labs_per_day = self.config.get('max_labs_per_day', 3)  # Default 3, configurable
                    if current_lab_count >= max_labs_per_day:
                        continue  # Skip: too many labs on this day already
                
                # CRITICAL VALIDATION: Check if placing here would create consecutive blanks
                # Get current occupied periods for this group on this day
                current_day_periods = sorted([slot_by_id[sid].period for sid in used_group_slot[session.student_group] 
                                             if slot_by_id.get(sid) and slot_by_id[sid].day == day])
                
                # Simulate adding this slot
                simulated_periods = sorted(current_day_periods + [slot.period])
                
                # Check for consecutive blanks in the range of occupied periods
                if len(simulated_periods) >= 2:
                    min_occupied = simulated_periods[0]
                    max_occupied = simulated_periods[-1]
                    
                    # Check all gaps between occupied periods
                    has_consecutive_blanks = False
                    for i in range(len(simulated_periods) - 1):
                        gap_start = simulated_periods[i]
                        gap_end = simulated_periods[i + 1]
                        gap_size = gap_end - gap_start - 1
                        
                        # If gap has 2+ consecutive blank periods, this is invalid
                        if gap_size >= 2:
                            has_consecutive_blanks = True
                            break
                    
                    # Skip this slot if it would create or maintain consecutive blanks
                    if has_consecutive_blanks:
                        continue

                # Try faculty in order (already sorted by workload)
                for fac in cand_faculty:
                    if slot_id not in faculty_avail.get(fac.id, set()):
                        continue
                    if slot_id in used_faculty_slot[fac.id]:
                        continue

                    # Find a free eligible room for this slot
                    taken = used_room_slot[slot_id]
                    room_found = None
                    for r in eligible_rooms:
                        if r.id not in taken:
                            room_found = r
                            break
                    if not room_found:
                        continue

                    # Place assignment
                    assignments.append({
                        "session_id": session.id,
                        "faculty_id": fac.id,
                        "room_id": room_found.id,
                        "slot_id": slot_id,
                        "group": session.student_group,
                        "course_id": course.id,
                        "course_code": session.course_code,
                        "is_lab": session.is_lab,
                    })
                    used_faculty_slot[fac.id].add(slot_id)
                    used_group_slot[session.student_group].add(slot_id)
                    used_room_slot[slot_id].add(room_found.id)
                    group_day_count[session.student_group][day] += 1
                    faculty_hours[fac.id] += 1
                    
                    # Update new constraint trackers
                    group_day_courses[session.student_group][day].add(course.id)
                    group_day_period_course[session.student_group][day][slot.period] = course.id
                    if session.is_lab:
                        group_day_lab_count[session.student_group][day] += 1
                    
                    placed = True
                    break  # OPTIMIZATION: Early exit from faculty loop

                if placed:
                    break  # OPTIMIZATION: Early exit from slot loop

            if not placed:
                # Provide more detailed warning about why placement failed
                reason = "unknown reason"
                if session.is_lab and group_day_lab_count[session.student_group].get(day, 0) >= self.config.get('max_labs_per_day', 3):
                    reason = "max labs per day limit reached"
                elif course.id in group_day_courses[session.student_group].get(day, set()):
                    reason = "same course already scheduled today (avoiding repetition)"
                warnings.append(f"⚠️ Could not place session for course {course.code} (group {session.student_group}) - {reason}")

        greedy_time = time.time() - greedy_start
        
        # Validate constraints after generation
        constraint_violations = self._validate_schedule_constraints(assignments, context)
        if constraint_violations:
            for violation in constraint_violations:
                warnings.append(f"⚠️ Constraint violation: {violation}")
        
        if not assignments:
            return {
                "success": False, 
                "error": "Greedy scheduler could not place any sessions.", 
                "warnings": warnings,
                "greedy_time": greedy_time
            }

        return {
            "success": True, 
            "assignments": assignments, 
            "warnings": warnings,
            "greedy_time": greedy_time,
            "placement_rate": len(assignments) / len(sessions) if sessions else 0
        }

    def _validate_schedule_constraints(self, assignments, context):
        """Validate that the generated schedule satisfies all constraints"""
        violations = []
        slot_by_id = context["slot_by_id"]
        max_labs_per_day = self.config.get('max_labs_per_day', 3)
        
        # Group assignments by student group and day
        group_day_schedule = defaultdict(lambda: defaultdict(list))
        for assignment in assignments:
            slot = slot_by_id.get(assignment["slot_id"])
            if slot:
                group_day_schedule[assignment["group"]][slot.day].append({
                    "period": slot.period,
                    "course_id": assignment["course_id"],
                    "course_code": assignment["course_code"],
                    "is_lab": assignment["is_lab"]
                })
        
        # Check constraints for each group and day
        for group, days in group_day_schedule.items():
            for day, schedule in days.items():
                # Sort by period
                schedule.sort(key=lambda x: x["period"])
                
                # Check 1: No consecutive same lectures
                for i in range(len(schedule) - 1):
                    if schedule[i]["course_id"] == schedule[i+1]["course_id"] and schedule[i]["period"] + 1 == schedule[i+1]["period"]:
                        violations.append(f"{group} on {day}: Consecutive {schedule[i]['course_code']} lectures (periods {schedule[i]['period']}-{schedule[i+1]['period']})")
                
                # Check 2: Count labs per day
                lab_count = sum(1 for item in schedule if item["is_lab"])
                if lab_count > max_labs_per_day:
                    violations.append(f"{group} on {day}: {lab_count} labs scheduled (max: {max_labs_per_day})")
                
                # Check 3: Same course scheduled multiple times in a day (for labs)
                course_counts = defaultdict(int)
                for item in schedule:
                    if item["is_lab"]:
                        course_counts[item["course_code"]] += 1
                
                for course_code, count in course_counts.items():
                    if count > 1:
                        violations.append(f"{group} on {day}: {course_code} lab scheduled {count} times")
        
        return violations

    # --------------------------------------------------------------------- #
    # Context Preparation
    # --------------------------------------------------------------------- #
    def _load_context(self, filters=None):
        filters = filters or {}
        
        # Use pre-filtered data if provided in __init__, otherwise query from DB
        if self.courses is not None:
            courses = self.courses
        else:
            course_query = Course.query
            if filters.get('program'):
                course_query = course_query.filter_by(program=filters['program'])
            if filters.get('semester'):
                try:
                    sem = int(filters['semester'])
                    course_query = course_query.filter_by(semester=sem)
                except (ValueError, TypeError):
                    pass
            courses = course_query.all()
        
        if self.student_groups is not None:
            # Use explicitly provided groups (even if empty list)
            student_groups = self.student_groups
            if self.verbose:
                print(f"[LOAD_CONTEXT] Using {len(student_groups)} explicitly provided groups")
        else:
            # Query groups from database
            group_query = StudentGroup.query
            if filters.get('program'):
                group_query = group_query.filter_by(program=filters['program'])
            if filters.get('semester'):
                try:
                    sem = int(filters['semester'])
                    group_query = group_query.filter_by(semester=sem)
                except (ValueError, TypeError):
                    pass
            student_groups = group_query.all()
            if self.verbose:
                print(f"[LOAD_CONTEXT] Queried {len(student_groups)} groups from database")
        
        faculty = Faculty.query.all()
        rooms = Room.query.all()
        time_slots = TimeSlot.query.order_by(TimeSlot.day, TimeSlot.period).all()

        # CROSS-DEPARTMENT WORKLOAD TRACKING:
        # Load existing timetable entries to prevent overlaps and track total workload
        # This is critical for faculty teaching in multiple departments
        existing_assignments = TimetableEntry.query.all()
        
        # Build maps of existing assignments
        faculty_existing_slots = defaultdict(set)  # faculty_id -> set of slot_ids already assigned
        faculty_existing_hours = defaultdict(int)  # faculty_id -> total hours already assigned
        
        for entry in existing_assignments:
            faculty_existing_slots[entry.faculty_id].add(entry.time_slot_id)
            faculty_existing_hours[entry.faculty_id] += 1
        
        if self.verbose and existing_assignments:
            print(f"[LOAD_CONTEXT] Loaded {len(existing_assignments)} existing timetable entries")
            print(f"[LOAD_CONTEXT] Faculty with existing assignments: {len(faculty_existing_hours)}")
            for fac_id, hours in list(faculty_existing_hours.items())[:5]:  # Show first 5
                fac = next((f for f in faculty if f.id == fac_id), None)
                if fac:
                    print(f"[LOAD_CONTEXT]   - {fac.name}: {hours} hours already assigned")

        # Read period configuration to allow per-group/day maximums
        period_config = PeriodConfig.query.first() if 'PeriodConfig' in globals() else None
        if period_config:
            max_per_day_for_group = getattr(period_config, 'max_periods_per_day_per_group', period_config.periods_per_day)
        else:
            max_per_day_for_group = 0

        # Only create Default group if no groups exist AND groups were queried (not explicitly provided)
        if not student_groups and self.student_groups is None:
            if self.verbose:
                print("[LOAD_CONTEXT] No groups found, creating Default group")
            default_group = StudentGroup(name="Default", description="Auto-generated group")
            self.db.session.add(default_group)
            self.db.session.commit()
            student_groups = [default_group]
        elif not student_groups:
            if self.verbose:
                print("[LOAD_CONTEXT] WARNING: No groups provided and none found in database!")

        slot_by_id = {slot.id: slot for slot in time_slots}
        slots_by_day: Dict[str, List[TimeSlot]] = defaultdict(list)
        for slot in time_slots:
            slots_by_day[slot.day].append(slot)
        for day in slots_by_day:
            slots_by_day[day].sort(key=lambda s: s.period)

        faculty_availability = self._build_faculty_availability_map(faculty, slot_by_id)
        faculty_expertise = self._build_faculty_expertise_map(faculty)
        
        # Enhanced: Track faculty seniority (for constraint 6)
        faculty_seniority = self._estimate_faculty_seniority(faculty)

        sessions = self._build_sessions(courses, student_groups)
        room_capabilities = self._build_room_capabilities(rooms)
        
        # OPTIMIZATION: Pre-compute eligibility maps once (cached for O(1) lookup)
        # This eliminates N+1 queries during assignment phase
        course_faculty_cache = {}
        course_room_cache = {}
        for course in courses:
            course_faculty_cache[course.id] = self._faculty_for_course(course, faculty, faculty_expertise)
            course_room_cache[course.id] = self._rooms_for_course(course, rooms, room_capabilities)

        return {
            "courses": courses,
            "course_by_id": {course.id: course for course in courses},
            "faculty": faculty,
            "faculty_by_id": {f.id: f for f in faculty},
            "rooms": rooms,
            "time_slots": time_slots,
            "slot_by_id": slot_by_id,
            "slots_by_day": slots_by_day,
            "student_groups": student_groups,
            "sessions": sessions,
            "faculty_availability": faculty_availability,
            "faculty_expertise": faculty_expertise,
            "faculty_seniority": faculty_seniority,
            "max_periods_per_day_per_group": max_per_day_for_group,
            "room_capabilities": room_capabilities,
            # OPTIMIZATION: Pre-computed caches
            "course_faculty_cache": course_faculty_cache,
            "course_room_cache": course_room_cache,
            # CROSS-DEPARTMENT TRACKING: Existing assignments
            "faculty_existing_slots": faculty_existing_slots,
            "faculty_existing_hours": faculty_existing_hours,
        }

    def _estimate_faculty_seniority(self, faculty_list: List[Faculty]) -> Dict[int, float]:
        """
        Constraint 6: Estimate faculty seniority based on workload patterns.
        Higher max_hours_per_week typically indicates senior faculty.
        Returns score 0-1 where 1 is most senior.
        """
        seniority = {}
        if not faculty_list:
            return seniority
        
        # Use max_hours as proxy for seniority (senior faculty often have higher max)
        max_hours_values = [f.max_hours_per_week or 16 for f in faculty_list]
        max_val = max(max_hours_values) if max_hours_values else 16
        min_val = min(max_hours_values) if max_hours_values else 4
        
        for faculty in faculty_list:
            max_h = faculty.max_hours_per_week or 16
            if max_val == min_val:
                seniority[faculty.id] = 0.5
            else:
                # Normalize: higher max_hours = higher seniority
                seniority[faculty.id] = (max_h - min_val) / (max_val - min_val)
        
        return seniority
# branch
    def _build_faculty_availability_map(self, faculty_list: List[Faculty], slot_by_id: Dict[int, TimeSlot]):
        """Constraint 3: Build availability map with preference scoring"""
        availability = {}
        for faculty in faculty_list:
            if not faculty.availability:
                availability[faculty.id] = set(slot_by_id.keys())
                continue
            raw_avail = faculty.availability
            # Normalize availability into a dict: if malformed -> allow all slots
            if isinstance(raw_avail, dict):
                availability_json = raw_avail
            else:
                try:
                    if not isinstance(raw_avail, str):
                        # Non-string (float/int/None) -> treat as empty
                        raise TypeError('Non-string availability payload')
                    availability_json = json.loads(raw_avail) if raw_avail.strip() else {}
                except (json.JSONDecodeError, TypeError, ValueError):
                    availability[faculty.id] = set(slot_by_id.keys())
                    continue

            allowed_slots: Set[int] = set()
            for day, periods in availability_json.items():
                if not isinstance(periods, (list, tuple)):
                    periods = periods.get("periods", [])
                normalized_periods = set()
                for period in periods:
                    if isinstance(period, dict) and "period" in period:
                        normalized_periods.add(int(period["period"]))
                    else:
                        try:
                            normalized_periods.add(int(period))
                        except (TypeError, ValueError):
                            continue

                for slot in slot_by_id.values():
                    if slot.day.lower() == day.lower() and slot.period in normalized_periods:
                        allowed_slots.add(slot.id)

            availability[faculty.id] = allowed_slots if allowed_slots else set(slot_by_id.keys())

        return availability

    def _build_faculty_expertise_map(self, faculty_list: List[Faculty]):
        """Constraint 4 & 8: Build expertise map for multi-course handling"""
        expertise_map: Dict[int, Set[str]] = {}
        for faculty in faculty_list:
            if faculty.expertise:
                items = [code.strip().lower() for code in faculty.expertise.split(",") if code.strip()]
                expertise_map[faculty.id] = set(items)
            else:
                expertise_map[faculty.id] = set()
        return expertise_map

    def _build_room_capabilities(self, rooms: List[Room]):
        """Constraint 2: Build room capabilities with lab tagging"""
        capability_map: Dict[int, Set[str]] = {}
        for room in rooms:
            tags = set()
            if room.tags:
                tags.update(tag.strip().lower() for tag in room.tags.split(",") if tag.strip())
            if room.room_type == "lab":
                tags.add("lab")
            capability_map[room.id] = tags
        return capability_map

    def _eligible_groups_for_course(self, course: Course, groups: List[StudentGroup]):
        """
        Determines which student groups should take a specific course.
        Enforces strict hierarchy: Program -> Branch -> Semester.
        OPTIMIZED: Pre-normalize course attributes once, cache group attributes.
        """
        eligible = []
        
        # OPTIMIZATION: Get and normalize course attributes ONCE before loop
        c_prog = getattr(course, 'program', None)
        c_sem = getattr(course, 'semester', None)
        c_branch = getattr(course, 'branch', None)
        
        # Pre-normalize course attributes (done once, not per-group)
        if c_prog:
            c_prog = str(c_prog).strip().lower()
        if c_sem is not None:
            try:
                c_sem = int(c_sem)
            except (ValueError, TypeError):
                c_sem = None
        if c_branch:
            c_branch = str(c_branch).strip().lower()
        
        # Check if course has any constraints
        has_constraints = bool(c_prog or c_sem is not None or c_branch)
        
        # If course has no constraints, match to all groups (common course)
        if not has_constraints:
            return groups

        # OPTIMIZATION: Pre-normalize all groups once (cache for reuse)
        # Build normalized group cache if not exists
        if not hasattr(self, '_group_cache'):
            self._group_cache = {}
        
        # Course has constraints - match groups that satisfy them
        for group in groups:
            # Skip "Default" group if course has specific constraints
            if group.name.lower() == "default" and has_constraints:
                continue
            
            # OPTIMIZATION: Use cached normalized group attributes
            group_id = id(group)  # Use object id as cache key
            if group_id not in self._group_cache:
                # Normalize and cache group attributes
                g_prog = getattr(group, 'program', None)
                g_sem = getattr(group, 'semester', None)
                g_branch = getattr(group, 'branch', None)
                
                # Normalize
                if g_prog:
                    g_prog = str(g_prog).strip().lower()
                if g_sem is not None:
                    try:
                        g_sem = int(g_sem)
                    except (ValueError, TypeError):
                        g_sem = None
                if g_branch:
                    g_branch = str(g_branch).strip().lower()
                
                self._group_cache[group_id] = (g_prog, g_sem, g_branch)
            else:
                g_prog, g_sem, g_branch = self._group_cache[group_id]
            
            # 1. Program Match: If course has program, group must match (or group has no program)
            if c_prog:
                if g_prog and c_prog != g_prog:
                    continue
            
            # 2. Semester Match: If course has semester, group must match (or group has no semester)
            if c_sem is not None:
                if g_sem is not None and c_sem != g_sem:
                    continue
            
            # 3. Branch Match: If course has branch, group must match (or group has no branch)
            if c_branch:
                if g_branch and c_branch != g_branch:
                    continue
            
            # All constraints satisfied (or group doesn't have conflicting constraints)
            eligible.append(group)
        
        # If no groups matched but course has constraints, log warning
        if not eligible and has_constraints and self.verbose:
            print(f"[MATCH] Course {course.code} (P:{c_prog}, S:{c_sem}, B:{c_branch}) matched 0 groups")
            # Show what groups we have for debugging
            if groups:
                print(f"[MATCH] Available groups (logging first 3):")
                for g in groups[:3]: 
                    g_name = getattr(g, 'name', '?')
                    # Try to get from cache first
                    g_id = getattr(g, 'id', None)
                    cache_info = ""
                    if hasattr(self, '_group_cache') and g_id in self._group_cache:
                        gp, gs, gb = self._group_cache[g_id]
                        cache_info = f" [CACHE: P:{gp}, S:{gs}, B:{gb}]"
                    
                    # Direct attributes
                    p = getattr(g, 'program', 'None')
                    b = getattr(g, 'branch', 'None')
                    s = getattr(g, 'semester', 'None')
                    print(f"[MATCH]   - Group '{g_name}' (Raw: P:{p}, B:{b}, S:{s}){cache_info}")
        
        return eligible

    def _build_sessions(self, courses: List[Course], groups: List[StudentGroup]):
        sessions = []
        session_id = 1
        
        # Skip verbose tracking in fast mode
        if self.verbose:
            group_usage = defaultdict(int)
            print(f"[BUILD_SESSIONS] Building sessions for {len(courses)} courses and {len(groups)} groups")
        
        for course in courses:
            eligible_groups = self._eligible_groups_for_course(course, groups)
            
            if self.verbose and not eligible_groups:
                print(f"[BUILD_SESSIONS] WARNING: Course {course.code} matched 0 groups!")
            
            is_lab = course.course_type == "practical"
            hours = course.hours_per_week
            
            for group in eligible_groups:
                if self.verbose:
                    group_usage[group.name] += 1
                # Pre-compute course code lowercase once
                course_code_lower = course.code.lower()
                # Create all sessions for this course-group pair in one go
                for _ in range(hours):
                    sessions.append(
                        Session(
                            id=session_id,
                            course_id=course.id,
                            course_code=course_code_lower,
                            course_type=course.course_type,
                            student_group=group.name,
                            is_lab=is_lab,
                        )
                    )
                    session_id += 1
        
        if self.verbose:
            print(f"[BUILD_SESSIONS] Created {len(sessions)} sessions")
            print(f"[BUILD_SESSIONS] Group usage: {dict(group_usage)}")
        
        return sessions

    # --------------------------------------------------------------------- #
    # Bound Analyzer (Constraint 1)
    # --------------------------------------------------------------------- #
    def _run_bound_analyzer(self, context):
        """Enhanced bound analysis with cross-department workload validation"""
        warnings = []
        total_session_hours = len(context["sessions"])
        
        # Get existing assignments for cross-department tracking
        faculty_existing_hours = context.get("faculty_existing_hours", defaultdict(int))
        
        # CROSS-DEPARTMENT WORKLOAD VALIDATION:
        # Calculate total workload across ALL departments and semesters
        # This prevents over-scheduling faculty who teach in multiple departments
        
        # Constraint 1: Validate faculty workload bounds (including existing assignments)
        total_faculty_capacity = 0
        total_faculty_minimum = 0
        faculty_remaining_capacity = {}
        
        for f in context["faculty"]:
            existing_hours = faculty_existing_hours.get(f.id, 0)
            max_hours = f.max_hours_per_week or 16
            min_hours = f.min_hours_per_week or 0
            
            # Calculate remaining capacity after existing assignments
            remaining = max_hours - existing_hours
            faculty_remaining_capacity[f.id] = max(0, remaining)
            
            total_faculty_capacity += remaining
            total_faculty_minimum += max(0, min_hours - existing_hours)
            
            # Warn if faculty is already at or over capacity
            if existing_hours >= max_hours:
                warnings.append(
                    f"⚠️ {f.name} already assigned {existing_hours}h (max: {max_hours}h) - no capacity for new sessions"
                )
            elif existing_hours > 0:
                if self.verbose:
                    print(f"[BOUND_ANALYZER] {f.name}: {existing_hours}h assigned, {remaining}h remaining (max: {max_hours}h)")
        
        if total_session_hours > total_faculty_capacity:
            warnings.append(
                f"⚠️ Workload Issue: New sessions ({total_session_hours}h) exceed remaining faculty capacity ({total_faculty_capacity}h)"
            )
            warnings.append(
                f"   Total existing assignments: {sum(faculty_existing_hours.values())}h across all departments"
            )
            return {"feasible": False, "warnings": warnings}
        
        if total_session_hours < total_faculty_minimum:
            warnings.append(
                f"ℹ️ Under-utilization: Total sessions ({total_session_hours}h) below faculty minimum ({total_faculty_minimum}h)"
            )

        # Constraint 2: Validate lab availability
        lab_sessions = [s for s in context["sessions"] if s.is_lab]
        lab_rooms = [r for r in context["rooms"] if r.room_type == "lab"]
        lab_capacity = len(lab_rooms) * len(context["time_slots"])
        
        if len(lab_sessions) > lab_capacity:
            warnings.append(
                f"⚠️ Lab Shortage: {len(lab_sessions)} lab sessions need {len(lab_rooms)} labs × {len(context['time_slots'])} slots"
            )
            return {"feasible": False, "warnings": warnings}

        # Constraint 3: Validate faculty availability coverage (accounting for existing assignments)
        faculty_existing_slots = context.get("faculty_existing_slots", defaultdict(set))
        
        for faculty in context["faculty"]:
            available_slots = context["faculty_availability"].get(faculty.id, set())
            existing_slots = faculty_existing_slots.get(faculty.id, set())
            
            # Remove already-used slots from available slots
            remaining_slots = available_slots - existing_slots
            remaining_slot_count = len(remaining_slots)
            
            existing_hours = faculty_existing_hours.get(faculty.id, 0)
            min_hours = faculty.min_hours_per_week or 0
            remaining_min_hours = max(0, min_hours - existing_hours)
            
            if remaining_slot_count < remaining_min_hours:
                warnings.append(
                    f"⚠️ {faculty.name} has only {remaining_slot_count} free slots but needs {remaining_min_hours} more hours (already assigned: {existing_hours}h)"
                )

        # Enhanced check: ensure each faculty has enough possible session assignments
        # (considering expertise, eligible rooms, availability, and existing assignments)
        sessions = context.get("sessions", [])
        for faculty in context["faculty"]:
            possible_session_count = 0
            avail_slots = context["faculty_availability"].get(faculty.id, set())
            existing_slots = faculty_existing_slots.get(faculty.id, set())
            remaining_slots = avail_slots - existing_slots
            
            for session in sessions:
                course = context["course_by_id"].get(session.course_id)
                if not course:
                    continue
                # Check faculty expertise eligibility
                eligible_faculty = self._faculty_for_course(course, [faculty], context.get("faculty_expertise", {}))
                if not eligible_faculty:
                    continue

                # Check for at least one eligible room for this course
                eligible_rooms = self._rooms_for_course(course, context.get("rooms", []), context.get("room_capabilities", {}))
                if not eligible_rooms:
                    continue

                # Check if there exists at least one timeslot where faculty is available AND not already assigned
                if any(slot.id in remaining_slots for slot in context.get("time_slots", [])):
                    possible_session_count += 1

            existing_hours = faculty_existing_hours.get(faculty.id, 0)
            min_hours = faculty.min_hours_per_week or 0
            remaining_min_hours = max(0, min_hours - existing_hours)
            
            if possible_session_count < remaining_min_hours:
                warnings.append(
                    f"⚠️ Feasibility Issue: {faculty.name} can teach at most {possible_session_count} new sessions but needs {remaining_min_hours} more (already assigned: {existing_hours}h, min: {min_hours}h)"
                )
                # Do not fail here: report as a warning so that soft-minima in the ILP
                # can be used to find a best-effort solution. Returning infeasible would
                # prevent the ILP from running even when we allow shortfalls.

        return {"feasible": True, "warnings": warnings}

    # --------------------------------------------------------------------- #
    # ILP Solver (Constraints 1-3)
    # --------------------------------------------------------------------- #
    def _solve_with_ilp(self, context):
        """Enhanced ILP with lab priority and availability focus.
        In fast mode, uses a reduced formulation (session, faculty, slot) and assigns rooms greedily after solving.
        """
        if self.fast_mode:
            return self._solve_with_ilp_fast(context)
        
        # Full formulation (includes rooms in decision variables)
        warnings = []
        problem = pulp.LpProblem("Timetable", pulp.LpMinimize)
        
        if self.verbose:
            print(f"[ILP] Starting ILP solver with {len(context['sessions'])} sessions")
            print(f"[ILP] Faculty count: {len(context['faculty'])}")
            print(f"[ILP] Room count: {len(context['rooms'])}")
            print(f"[ILP] Time slot count: {len(context['time_slots'])}")
        
        # Build candidates for each session
        session_candidates = {}
        decision_vars = {}
        
        for session in context["sessions"]:
            course = context["course_by_id"][session.course_id]
            eligible_faculty = self._faculty_for_course(course, context["faculty"], context["faculty_expertise"])
            eligible_rooms = self._rooms_for_course(course, context["rooms"], context["room_capabilities"])
            
            if self.verbose:
                print(f"[ILP] Session {session.id} ({course.code}): {len(eligible_faculty)} faculty, {len(eligible_rooms)} rooms")
            
            if not eligible_faculty:
                warnings.append(f"⚠️ No faculty available for course {course.code}")
                if self.verbose:
                    print(f"[ILP] WARNING: No eligible faculty for {course.code}")
                continue
            if not eligible_rooms:
                warnings.append(f"⚠️ No suitable rooms for course {course.code}")
                if self.verbose:
                    print(f"[ILP] WARNING: No eligible rooms for {course.code}")
                continue
            
            # CROSS-DEPARTMENT: Get existing assignments to exclude
            faculty_existing_slots = context.get("faculty_existing_slots", defaultdict(set))
            
            candidates = []
            for faculty in eligible_faculty:
                # Constraint 3: Only consider available timeslots
                available_slots = context["faculty_availability"].get(faculty.id, set())
                
                # CROSS-DEPARTMENT HARD CONSTRAINT:
                # Exclude slots where faculty is already assigned (from other departments/semesters)
                existing_slots = faculty_existing_slots.get(faculty.id, set())
                truly_available_slots = available_slots - existing_slots
                
                if self.verbose:
                    total_avail = len(available_slots)
                    existing_count = len(existing_slots)
                    remaining_count = len(truly_available_slots)
                    if existing_count > 0:
                        print(f"[ILP]   Faculty {faculty.name} (ID:{faculty.id}): {total_avail} available, {existing_count} already assigned, {remaining_count} free")
                    else:
                        print(f"[ILP]   Faculty {faculty.name} (ID:{faculty.id}) has {remaining_count} available slots")
                
                for room in eligible_rooms:
                    for slot in context["time_slots"]:
                        # Skip if faculty not available
                        if slot.id not in truly_available_slots:
                            continue
                        
                        var_name = f"s{session.id}_f{faculty.id}_r{room.id}_t{slot.id}"
                        var = pulp.LpVariable(var_name, cat="Binary")
                        decision_vars[var_name] = var
                        
                        # Constraint 2 & 6: Calculate priority score
                        priority_score = 0
                        if session.is_lab:
                            priority_score += self.lab_priority_weight
                        
                        # Constraint 6: Prefer early/preferred slots for senior faculty
                        if self.senior_faculty_preference:
                            seniority = context["faculty_seniority"].get(faculty.id, 0.5)
                            if seniority > 0.7 and slot.period <= 3:  # Morning slots for senior
                                priority_score -= 10
                        
                        candidates.append({
                            "var": var,
                            "faculty_id": faculty.id,
                            "room_id": room.id,
                            "slot_id": slot.id,
                            "group": session.student_group,
                            "course_id": course.id,
                            "course_code": session.course_code,
                            "is_lab": session.is_lab,
                            "priority": priority_score
                        })
            
            if not candidates:
                warnings.append(f"⚠️ No valid candidates for session {session.id} of {course.code}")
                if self.verbose:
                    print(f"[ILP] ERROR: No valid candidates for session {session.id} ({course.code}) - likely no faculty availability!")
                continue
            if self.verbose:
                print(f"[ILP] Session {session.id} has {len(candidates)} valid candidates")
            session_candidates[session.id] = candidates
            
            # Constraint: Each session assigned exactly once
            # If `maximize_fill` config is set, allow session to be unassigned (<=1)
            if self.config.get('maximize_fill', False):
                problem += pulp.lpSum(c["var"] for c in candidates) <= 1, f"session_{session.id}_opt"
            else:
                problem += pulp.lpSum(c["var"] for c in candidates) == 1, f"session_{session.id}"
        
        if self.verbose:
            print(f"[ILP] Total sessions with candidates: {len(session_candidates)} out of {len(context['sessions'])}")
        
        # GLOBAL CONSTRAINT: No faculty/room/group conflicts per timeslot
        # This ensures faculty can only be assigned to ONE session per slot across ALL departments
        faculty_slot_usage = defaultdict(list)
        room_slot_usage = defaultdict(list)
        group_slot_usage = defaultdict(list)
        
        for candidates in session_candidates.values():
            for candidate in candidates:
                faculty_slot_usage[(candidate["faculty_id"], candidate["slot_id"])].append(candidate["var"])
                room_slot_usage[(candidate["room_id"], candidate["slot_id"])].append(candidate["var"])
                group_slot_usage[(candidate["group"], candidate["slot_id"])].append(candidate["var"])
        
        # HARD CONSTRAINT: Faculty can teach at most 1 session per slot
        # Combined with the exclusion of existing_slots above, this prevents cross-department conflicts
        for key, vars_list in faculty_slot_usage.items():
            problem += pulp.lpSum(vars_list) <= 1, f"faculty_{key[0]}_slot_{key[1]}"
        for key, vars_list in room_slot_usage.items():
            problem += pulp.lpSum(vars_list) <= 1, f"room_{key[0]}_slot_{key[1]}"
        for key, vars_list in group_slot_usage.items():
            problem += pulp.lpSum(vars_list) <= 1, f"group_{key[0]}_slot_{key[1]}"

        # Constraint: limit total periods per group per day (configurable)
        max_per_day = context.get('max_periods_per_day_per_group', 0) or None
        if max_per_day is not None:
            # For each group + day, ensure assigned vars do not exceed the maximum
            for group in context.get('student_groups', []):
                for day, slots in context.get('slots_by_day', {}).items():
                    day_vars = []
                    slot_ids = {s.id for s in slots}
                    for candidates in session_candidates.values():
                        for c in candidates:
                            if c['group'] == group.name and c['slot_id'] in slot_ids:
                                day_vars.append(c['var'])
                    if day_vars:
                        problem += pulp.lpSum(day_vars) <= max_per_day, f"group_{group.name}_day_{day}_max"
        
        # Constraint 1: Faculty workload bounds
        faculty_hours = defaultdict(list)
        for candidates in session_candidates.values():
            for candidate in candidates:
                faculty_hours[candidate["faculty_id"]].append(candidate["var"])
        
        for faculty in context["faculty"]:
            if faculty.id in faculty_hours:
                total = pulp.lpSum(faculty_hours[faculty.id])
                # Make minimum-hours a soft constraint using a non-negative slack variable
                slack_name = f"slack_faculty_{faculty.id}"
                slack_var = pulp.LpVariable(slack_name, lowBound=0, cat="Continuous")
                problem += total + slack_var >= faculty.min_hours_per_week, f"faculty_{faculty.id}_min_soft"
                # Keep maximum as a hard constraint
                problem += total <= faculty.max_hours_per_week, f"faculty_{faculty.id}_max"
                # Store slack var on the faculty object for objective construction
                faculty._min_slack_var = slack_var
        
        # Constraint 2: At least one lab per student group
        for group in context["student_groups"]:
            lab_vars = []
            for candidates in session_candidates.values():
                for candidate in candidates:
                    if candidate["is_lab"] and candidate["group"] == group.name:
                        lab_vars.append(candidate["var"])
            if lab_vars:
                problem += pulp.lpSum(lab_vars) >= 1, f"group_{group.name}_min_lab"
        
        # Objective: Penalize minimum-hours shortfall (slack) heavily, plus priority scores
        objective_terms = []
        slack_penalty = self.config.get('min_violation_penalty', 1000)
        for faculty in context["faculty"]:
            if hasattr(faculty, '_min_slack_var'):
                # Penalize any slack (hours shortfall) to prefer meeting minima when possible
                objective_terms.append(slack_penalty * faculty._min_slack_var)
        
        # Add priority scores to objective
        for candidates in session_candidates.values():
            for candidate in candidates:
                objective_terms.append(candidate["priority"] * candidate["var"])

        # If maximizing fill is enabled, add a negative reward for assigning any candidate
        # so that the minimization objective will try to assign as many sessions as possible
        if self.config.get('maximize_fill', False):
            assign_reward = -self.config.get('assign_reward', 50)
            for candidates in session_candidates.values():
                for candidate in candidates:
                    objective_terms.append(assign_reward * candidate["var"]) 
        
        problem += pulp.lpSum(objective_terms)
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=30, threads=2)
        status = problem.solve(solver)
        
        if status != pulp.LpStatusOptimal:
            return {
                "success": False,
                "error": f"ILP solver failed with status: {pulp.LpStatus[status]}",
                "warnings": warnings
            }
        
        # Extract assignments
        assignments = []
        for session_id, candidates in session_candidates.items():
            for candidate in candidates:
                if pulp.value(candidate["var"]) > 0.5:
                    assignments.append({
                        "session_id": session_id,
                        "faculty_id": candidate["faculty_id"],
                        "room_id": candidate["room_id"],
                        "slot_id": candidate["slot_id"],
                        "group": candidate["group"],
                        "course_id": candidate["course_id"],
                        "course_code": candidate["course_code"],
                        "is_lab": candidate["is_lab"],
                    })
        
        return {
            "success": True,
            "assignments": assignments,
            "warnings": warnings,
            "session_candidates": session_candidates,
        }

    def _solve_with_ilp_fast(self, context):
        """Reduced ILP: assign (session, faculty, slot) only; assign rooms greedily after.
        This drastically reduces the decision variable count and solves faster.
        CROSS-DEPARTMENT: Excludes existing assignments as hard unavailabilities.
        """
        warnings = []
        problem = pulp.LpProblem("TimetableFast", pulp.LpMinimize)

        if self.verbose:
            print(f"[ILP-FAST] Sessions: {len(context['sessions'])}, Faculty: {len(context['faculty'])}, Slots: {len(context['time_slots'])}")

        # CROSS-DEPARTMENT: Get existing assignments
        faculty_existing_slots = context.get("faculty_existing_slots", defaultdict(set))
        
        if self.verbose and faculty_existing_slots:
            total_existing = sum(len(slots) for slots in faculty_existing_slots.values())
            print(f"[ILP-FAST] Excluding {total_existing} existing faculty slot assignments")

        # Build candidates per session without room dimension
        session_candidates = {}
        decision_vars = {}

        for session in context["sessions"]:
            course = context["course_by_id"][session.course_id]
            eligible_faculty = self._faculty_for_course(course, context["faculty"], context["faculty_expertise"])
            if not eligible_faculty:
                warnings.append(f"⚠️ No faculty available for course {course.code}")
                continue

            candidates = []
            for faculty in eligible_faculty:
                available_slots = context["faculty_availability"].get(faculty.id, set())
                
                # CROSS-DEPARTMENT HARD CONSTRAINT:
                # Exclude slots where faculty is already assigned
                existing_slots = faculty_existing_slots.get(faculty.id, set())
                truly_available_slots = available_slots - existing_slots
                
                if not truly_available_slots:
                    continue
                    
                # Optional pruning: limit slots considered per session to reduce vars
                # Default is to consider every available slot for complete week coverage
                max_slots_limit = self.config.get('max_slots_per_session')
                if not max_slots_limit or max_slots_limit <= 0:
                    max_slots_limit = len(context["time_slots"])
                limited_slots = []
                for slot in context["time_slots"]:
                    if slot.id in truly_available_slots:  # Use truly_available_slots instead of available_slots
                        limited_slots.append(slot)
                        if len(limited_slots) >= max_slots_limit:
                            break

                for slot in limited_slots:
                    var_name = f"s{session.id}_f{faculty.id}_t{slot.id}"
                    var = pulp.LpVariable(var_name, cat="Binary")
                    decision_vars[var_name] = var

                    priority_score = 0
                    if session.is_lab:
                        priority_score += self.lab_priority_weight
                    if self.senior_faculty_preference:
                        seniority = context["faculty_seniority"].get(faculty.id, 0.5)
                        if seniority > 0.7 and slot.period <= 3:
                            priority_score -= 10

                    candidates.append({
                        "var": var,
                        "faculty_id": faculty.id,
                        "slot_id": slot.id,
                        "group": session.student_group,
                        "course_id": course.id,
                        "course_code": session.course_code,
                        "is_lab": session.is_lab,
                        "priority": priority_score
                    })

            if not candidates:
                warnings.append(f"⚠️ No valid candidates for session {session.id} of {course.code}")
                continue

            session_candidates[session.id] = candidates
            if self.config.get('maximize_fill', False):
                problem += pulp.lpSum(c["var"] for c in candidates) <= 1, f"session_{session.id}_opt"
            else:
                problem += pulp.lpSum(c["var"] for c in candidates) == 1, f"session_{session.id}"

        # No faculty conflicts per slot; no group conflicts per slot
        faculty_slot_usage = defaultdict(list)
        group_slot_usage = defaultdict(list)
        for candidates in session_candidates.values():
            for c in candidates:
                faculty_slot_usage[(c["faculty_id"], c["slot_id"])].append(c["var"])
                group_slot_usage[(c["group"], c["slot_id"])].append(c["var"])
        for key, vars_list in faculty_slot_usage.items():
            problem += pulp.lpSum(vars_list) <= 1, f"faculty_{key[0]}_slot_{key[1]}"
        for key, vars_list in group_slot_usage.items():
            problem += pulp.lpSum(vars_list) <= 1, f"group_{key[0]}_slot_{key[1]}"

        # Group per-day maximum
        max_per_day = context.get('max_periods_per_day_per_group', 0) or None
        if max_per_day is not None:
            for group in context.get('student_groups', []):
                for day, slots in context.get('slots_by_day', {}).items():
                    day_vars = []
                    slot_ids = {s.id for s in slots}
                    for candidates in session_candidates.values():
                        for c in candidates:
                            if c['group'] == group.name and c['slot_id'] in slot_ids:
                                day_vars.append(c['var'])
                    if day_vars:
                        problem += pulp.lpSum(day_vars) <= max_per_day, f"group_{group.name}_day_{day}_max"

        # Faculty workload bounds with slack
        faculty_hours = defaultdict(list)
        for candidates in session_candidates.values():
            for c in candidates:
                faculty_hours[c["faculty_id"]].append(c["var"])
        for faculty in context["faculty"]:
            if faculty.id in faculty_hours:
                total = pulp.lpSum(faculty_hours[faculty.id])
                slack = pulp.LpVariable(f"slack_faculty_{faculty.id}", lowBound=0, cat="Continuous")
                problem += total + slack >= faculty.min_hours_per_week, f"faculty_{faculty.id}_min_soft"
                problem += total <= faculty.max_hours_per_week, f"faculty_{faculty.id}_max"
                faculty._min_slack_var = slack

        # At least one lab per group (if any lab sessions exist for that group)
        for group in context["student_groups"]:
            lab_vars = []
            for candidates in session_candidates.values():
                for c in candidates:
                    if c["is_lab"] and c["group"] == group.name:
                        lab_vars.append(c["var"])
            if lab_vars:
                problem += pulp.lpSum(lab_vars) >= 1, f"group_{group.name}_min_lab"

        # Objective: penalize slack + use priorities; optionally reward assignment fill
        objective_terms = []
        slack_penalty = self.config.get('min_violation_penalty', 1000)
        for faculty in context["faculty"]:
            if hasattr(faculty, '_min_slack_var'):
                objective_terms.append(slack_penalty * faculty._min_slack_var)
        for candidates in session_candidates.values():
            for c in candidates:
                objective_terms.append(c["priority"] * c["var"])
        if self.config.get('maximize_fill', False):
            assign_reward = -self.config.get('assign_reward', 50)
            for candidates in session_candidates.values():
                for c in candidates:
                    objective_terms.append(assign_reward * c["var"])
        problem += pulp.lpSum(objective_terms)

        solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=20, threads=2)
        status = problem.solve(solver)
        if status != pulp.LpStatusOptimal:
            return {
                "success": False,
                "error": f"ILP-FAST solver failed with status: {pulp.LpStatus[status]}",
                "warnings": warnings
            }

        # Extract assignments (no rooms yet)
        assignments = []
        for session_id, candidates in session_candidates.items():
            for c in candidates:
                if pulp.value(c["var"]) > 0.5:
                    assignments.append({
                        "session_id": session_id,
                        "faculty_id": c["faculty_id"],
                        "slot_id": c["slot_id"],
                        "group": c["group"],
                        "course_id": c["course_id"],
                        "course_code": c["course_code"],
                        "is_lab": c["is_lab"],
                    })

        # Assign rooms greedily per slot
        room_warnings, assignments_with_rooms = self._assign_rooms_greedy(assignments, context)
        warnings.extend(room_warnings)
        return {
            "success": True,
            "assignments": assignments_with_rooms,
            "warnings": warnings,
            "session_candidates": session_candidates,
        }

    def _assign_rooms_greedy(self, assignments, context):
        """Greedy room assignment per slot; drop assignments that cannot get a valid room."""
        warnings = []
        result = []
        used_room_per_slot = defaultdict(set)
        rooms = context["rooms"]
        room_caps = context["room_capabilities"]

        # Index rooms by type for quick filtering
        rooms_by_type = {
            'lab': [r for r in rooms if r.room_type == 'lab'],
            'classroom': [r for r in rooms if r.room_type == 'classroom']
        }

        # Precompute eligible rooms for each course id
        eligible_rooms_cache = {}

        for a in assignments:
            course = context["course_by_id"].get(a["course_id"])
            if not course:
                continue
            key = course.id
            if key not in eligible_rooms_cache:
                # compute eligible rooms once
                eligible_rooms_cache[key] = self._rooms_for_course(course, rooms, room_caps)

            slot_id = a["slot_id"]
            taken = used_room_per_slot[slot_id]
            room_assigned = None
            for r in eligible_rooms_cache[key]:
                if r.id not in taken:
                    room_assigned = r
                    break
            if not room_assigned:
                warnings.append(f"⚠️ No available room for course {course.code} at slot {slot_id}; dropping this session.")
                continue
            used_room_per_slot[slot_id].add(room_assigned.id)
            b = a.copy()
            b["room_id"] = room_assigned.id
            result.append(b)

        return warnings, result

    def _faculty_for_course(self, course: Course, faculty_list: List[Faculty], expertise_map):
        """Constraint 4 & 8: Select faculty based on expertise"""
        course_code = course.code.lower()
        eligible = []
        for faculty in faculty_list:
            expertise = expertise_map.get(faculty.id, set())
            if not expertise or course_code in expertise:
                eligible.append(faculty)
        return eligible

    def _rooms_for_course(self, course: Course, rooms: List[Room], room_tags):
        """Constraint 2: Select rooms with required lab capabilities"""
        is_lab = course.course_type == "practical"
        required_tags = set()
        if course.required_room_tags:
            required_tags.update(tag.strip().lower() for tag in course.required_room_tags.split(",") if tag.strip())
        if is_lab:
            required_tags.add("lab")

        eligible = []
        for room in rooms:
            if is_lab and room.room_type != "lab":
                continue
            if not is_lab and room.room_type != "classroom":
                continue
            tags = room_tags.get(room.id, set())
            if required_tags and not required_tags.issubset(tags):
                continue
            eligible.append(room)
        return eligible

    # --------------------------------------------------------------------- #
    # Genetic Optimizer (Constraints 5-8)
    # --------------------------------------------------------------------- #
    def _refine_with_genetic_algorithm(self, context, base_assignments, session_candidates):
        """Enhanced GA with consecutive lecture prevention and multi-course handling"""
        if not base_assignments:
            return {"warnings": ["GA skipped – no ILP assignments to refine."]}

        population_size = 10
        generations = 15
        candidates_by_session = self._index_assignment_candidates(session_candidates)

        def clone(assignments):
            return [assignment.copy() for assignment in assignments]

        population = [clone(base_assignments)]
        while len(population) < population_size:
            mutated = self._mutate_assignment(population[0], candidates_by_session)
            population.append(mutated)

        for _ in range(generations):
            scored = sorted(
                [(self._fitness(individual, context), individual) for individual in population],
                key=lambda item: item[0],
            )
            population = [ind for _, ind in scored[:population_size // 2]]
            while len(population) < population_size:
                parents = self.random.sample(population[: max(1, len(population) // 2)], k=min(2, len(population)))
                child = self._crossover_assignments(*parents)
                child = self._mutate_assignment(child, candidates_by_session)
                population.append(child)

        best = min(population, key=lambda individual: self._fitness(individual, context))
        return {"assignments": best, "warnings": []}

    def _index_assignment_candidates(self, session_candidates):
        index = defaultdict(list)
        for session_id, candidates in session_candidates.items():
            for candidate in candidates:
                index[session_id].append({
                    "session_id": session_id,
                    "faculty_id": candidate["faculty_id"],
                    "room_id": candidate["room_id"],
                    "slot_id": candidate["slot_id"],
                    "group": candidate["group"],
                    "course_id": candidate["course_id"],
                    "course_code": candidate["course_code"],
                    "is_lab": candidate["is_lab"],
                })
        return index

    def _mutate_assignment(self, assignments, candidates_by_session):
        mutated = [assignment.copy() for assignment in assignments]
        if not mutated:
            return mutated
        target = self.random.choice(mutated)
        session_candidates = candidates_by_session.get(target["session_id"], [])
        if not session_candidates:
            return mutated
        replacement = self.random.choice(session_candidates)
        target.update(replacement)
        return mutated

    def _crossover_assignments(self, parent_a, parent_b=None):
        if parent_b is None:
            return [assignment.copy() for assignment in parent_a]
        child = []
        for a, b in zip(parent_a, parent_b):
            child.append(a.copy() if self.random.random() < 0.5 else b.copy())
        return child

    def _fitness(self, assignments, context):
        """Enhanced fitness with all constraint penalties"""
        penalty = 0
        faculty_hours = defaultdict(int)
        faculty_daily_hours = defaultdict(lambda: defaultdict(int))
        group_day_labs = defaultdict(int)
        group_daily_hours = defaultdict(lambda: defaultdict(int))
        slot_lookup = context["slot_by_id"]

        faculty_conflicts = set()
        room_conflicts = set()
        group_conflicts = set()

        # Build student group lookup for semester validation
        group_by_name = {g.name: g for g in context["student_groups"]}
        course_by_id = context["course_by_id"]

        for assignment in assignments:
            faculty_hours[assignment["faculty_id"]] += 1
            slot = slot_lookup[assignment["slot_id"]]
            faculty_daily_hours[assignment["faculty_id"]][slot.day] += 1
            group_daily_hours[assignment["group"]][slot.day] += 1
            if assignment["is_lab"]:
                group_day_labs[(assignment["group"], slot.day)] += 1

            faculty_key = (assignment["faculty_id"], slot.id)
            room_key = (assignment["room_id"], slot.id)
            group_key = (assignment["group"], slot.id)
            if faculty_key in faculty_conflicts or room_key in room_conflicts or group_key in group_conflicts:
                penalty += 100
            faculty_conflicts.add(faculty_key)
            room_conflicts.add(room_key)
            group_conflicts.add(group_key)
            
            # CRITICAL CONSTRAINT: Semester Matching Validation
            # This ensures courses are NEVER assigned to wrong semester groups
            course = course_by_id.get(assignment["course_id"])
            group = group_by_name.get(assignment["group"])
            
            if course and group:
                c_sem = getattr(course, 'semester', None)
                g_sem = getattr(group, 'semester', None)
                
                # If both course and group have semester defined, they MUST match
                # Exception: semester 0 or None means "open to all"
                if c_sem is not None and c_sem != 0 and g_sem is not None:
                    if int(c_sem) != int(g_sem):
                        # MASSIVE PENALTY - This should never happen
                        # This is a hard constraint violation
                        penalty += 10000
                        print(f"[CONSTRAINT VIOLATION] Course {course.code} (Semester {c_sem}) assigned to Group {group.name} (Semester {g_sem})")

        # Constraint 1: Workload bounds penalty
        for faculty in context["faculty"]:
            hours = faculty_hours.get(faculty.id, 0)
            if hours < faculty.min_hours_per_week:
                penalty += (faculty.min_hours_per_week - hours) * 15
            if hours > faculty.max_hours_per_week:
                penalty += (hours - faculty.max_hours_per_week) * 15

        # Constraint 2: Lab requirement penalty
        for group in context["student_groups"]:
            labs = sum(group_day_labs.get((group.name, day), 0) for day in context["slots_by_day"].keys())
            if labs == 0:
                penalty += 30

        # Constraint X: Group per-day maximum penalty
        max_per_day = context.get('max_periods_per_day_per_group', None)
        if max_per_day:
            for group_name, daily in group_daily_hours.items():
                for day, hours in daily.items():
                    if hours > max_per_day:
                        penalty += (hours - max_per_day) * 20

        # Constraint 5: Consecutive lecture penalty (major enhancement)
        penalty += self._consecutive_penalty(assignments, context) * self.consecutive_penalty_weight

        # Constraint 7: Daily balance penalty
        for faculty_id, daily_hours in faculty_daily_hours.items():
            for day, hours in daily_hours.items():
                if hours > 6:  # More than 6 hours in a day
                    penalty += (hours - 6) * 5

        return penalty

    def _consecutive_penalty(self, assignments, context):
        """Constraint 5: Heavily penalize consecutive lectures of same subject"""
        penalty = 0
        slot_lookup = context["slot_by_id"]
        grouped = defaultdict(list)
        
        for assignment in assignments:
            slot = slot_lookup[assignment["slot_id"]]
            key = (assignment["group"], assignment["course_code"], slot.day)
            grouped[key].append(slot)

        for slots in grouped.values():
            slots.sort(key=lambda s: s.period)
            for first, second in zip(slots, slots[1:]):
                if second.period == first.period + 1:
                    penalty += 10  # Heavy penalty for consecutive same subject
        
        return penalty

    # --------------------------------------------------------------------- #
    # Enhanced Features (Constraints 7, 9)
    # --------------------------------------------------------------------- #
    def _generate_faculty_schedules(self, assignments, context):
        """Constraint 7: Generate detailed per-faculty daily schedules"""
        schedules = defaultdict(lambda: defaultdict(list))
        slot_lookup = context["slot_by_id"]
        course_lookup = context["course_by_id"]
        
        for assignment in assignments:
            faculty_id = assignment["faculty_id"]
            slot = slot_lookup[assignment["slot_id"]]
            course = course_lookup[assignment["course_id"]]
            
            schedules[faculty_id][slot.day].append({
                "period": slot.period,
                "time": f"{slot.start_time}-{slot.end_time}",
                "course": course.code,
                "group": assignment["group"],
                "is_lab": assignment["is_lab"],
                "room_id": assignment["room_id"]
            })
        
        # Sort by period for each day
        for faculty_id in schedules:
            for day in schedules[faculty_id]:
                schedules[faculty_id][day].sort(key=lambda x: x["period"])
        
        return dict(schedules)

    def _detect_overwork(self, assignments, context):
        """Constraint 9: Detect and alert on faculty overwork (40+ hours)"""
        warnings = []
        faculty_hours = defaultdict(int)
        faculty_lookup = context["faculty_by_id"]
        
        for assignment in assignments:
            faculty_hours[assignment["faculty_id"]] += 1
        
        for faculty_id, hours in faculty_hours.items():
            if hours >= self.overwork_threshold:
                faculty = faculty_lookup.get(faculty_id)
                if faculty:
                    warnings.append(
                        f"🚨 OVERWORK ALERT: {faculty.name} assigned {hours} hours/week (threshold: {self.overwork_threshold}h) - Review workload!"
                    )
        
        return warnings

    # --------------------------------------------------------------------- #
    # Persistence
    # --------------------------------------------------------------------- #
    def _persist_assignments(self, assignments, context):
        import time
        start_time = time.time()
        print(f"[PERSIST] Starting to persist {len(assignments)} assignments...")
        
        # OPTIMIZATION: True bulk insert using MongoDB insert_many
        if assignments:
            coll_name = 'timetableentry'
            counters = self.db._db['__counters__']
            
            # Step 1: Pre-allocate all IDs in one database call
            res = counters.find_one_and_update(
                {'_id': coll_name}, 
                {'$inc': {'seq': len(assignments)}}, 
                upsert=True, 
                return_document=True
            )
            start_id = int(res['seq']) - len(assignments) + 1
            
            # Step 2: Build all documents in memory (no database calls)
            docs = []
            for i, assignment in enumerate(assignments):
                docs.append({
                    'id': start_id + i,
                    'course_id': assignment["course_id"],
                    'faculty_id': assignment["faculty_id"],
                    'room_id': assignment["room_id"],
                    'time_slot_id': assignment["slot_id"],
                    'student_group': assignment["group"],
                })
            
            # Step 3: Single bulk insert (parallel writes with ordered=False)
            if self.verbose:
                print(f"[PERSIST] Bulk inserting {len(docs)} entries...")
            bulk_start = time.time()
            self.db._db[coll_name].insert_many(docs, ordered=False)
            bulk_time = time.time() - bulk_start
            
            entries_created = len(docs)
            if self.verbose:
                print(f"[PERSIST] Successfully bulk inserted {entries_created} entries in {bulk_time:.2f}s!")
            
            # REMOVED: Verification query (trust bulk write)
            # Verification queries scan entire collection - very slow!
            
            total_time = time.time() - start_time
            if self.verbose:
                print(f"[PERSIST] Total persistence time: {total_time:.2f}s")
                print(f"[PERSIST] Average time per entry: {(total_time / len(docs) * 1000):.2f}ms")
        else:
            entries_created = 0
            if self.verbose:
                print(f"[PERSIST] No assignments to persist")
        
        return entries_created
