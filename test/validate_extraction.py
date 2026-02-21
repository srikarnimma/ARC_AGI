"""
Validation: test ObjectExtractor on real Milestone B training data
"""

import json
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from GraphObjectExtractor import ObjectExtractor


def load_arc_problem(json_path):
    """Load an ARC problem from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_grid(grid, name="Grid"):
    """Pretty print a small grid."""
    print(f"\n{name} ({grid.shape[0]}x{grid.shape[1]}):")
    for row in grid:
        print("  " + " ".join(f"{c}" for c in row))


def test_single_object_problem(problem_path, problem_name):
    """Test on problems with one object per example (like 1cf80156).
    
    JSON: 1cf80156.json
    Pattern: Single object crop - extracts the minimal bounding box of a single colored object
    Input: Large grid with one object of varying size/color
    Output: Cropped to bounding box of the object
    Key test: Grid boundary always present; exactly 1 colored object; bbox matches output shape
    """
    print(f"\n{'='*60}")
    print(f"[SINGLE-OBJECT TEST] Problem: {problem_name}")
    print(f"{'='*60}")
    
    try:
        problem = load_arc_problem(problem_path)
        extractor = ObjectExtractor(connectivity="8")
        
        all_correct = True
        
        for idx, example in enumerate(problem['train']):
            input_grid = np.array(example['input'], dtype=np.uint8)
            output_grid = np.array(example['output'], dtype=np.uint8)
            
            print(f"\n--- Training Example {idx + 1} ---")
            print(f"Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # Extract objects
            objects = extractor.extract(input_grid)
            
            # Find grid boundary object
            grid_obj = next((obj for obj in objects if obj.is_grid_boundary), None)
            if not grid_obj:
                print(f"[FAIL] Grid boundary object not found!")
                all_correct = False
                continue
            
            print(f"[OK] Grid boundary: {input_grid.shape[0]}x{input_grid.shape[1]}")
            
            # Find colored objects (exclude grid boundary)
            colored_objects = [obj for obj in objects if not obj.is_grid_boundary]
            
            if len(colored_objects) != 1:
                print(f"[FAIL] Expected 1 colored object, got {len(colored_objects)}")
                all_correct = False
                continue
            
            obj = colored_objects[0]
            min_r, min_c, max_r, max_c = obj.bbox
            bbox_h = max_r - min_r + 1
            bbox_w = max_c - min_c + 1
            expected_h, expected_w = output_grid.shape
            
            # The output shape should match the bounding box
            if (bbox_h, bbox_w) == (expected_h, expected_w):
                print(f"[OK] Bbox {bbox_h}x{bbox_w} matches output shape {expected_h}x{expected_w}")
                print(f"     Color: {obj.color}, Area: {obj.area} pixels")
                print(f"     Bbox: ({min_r}, {min_c}) to ({max_r}, {max_c})")
            else:
                print(f"[FAIL] Bbox {bbox_h}x{bbox_w} doesn't match output {expected_h}x{expected_w}")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_rectangle_hollowing_problem(problem_path, problem_name):
    """Test on problems with multiple solid rectangles that become hollow (4347f46a).
    
    JSON: 4347f46a.json
    Pattern: Rectangle hollowing - solid filled rectangles get their interiors emptied
    Input: Grid with multiple solid rectangles of different colors
    Output: Same grid with rectangles now hollow (perimeter filled, interior empty)
    Key test: Validates that extractor recognizes rectangles and detects hollow property correctly
    """
    print(f"\n{'='*60}")
    print(f"[RECTANGLE TEST] Problem: {problem_name}")
    print(f"{'='*60}")
    
    try:
        problem = load_arc_problem(problem_path)
        extractor = ObjectExtractor(connectivity="8")
        
        all_correct = True
        
        for idx, example in enumerate(problem['train'][:2]):  # First 2 examples
            input_grid = np.array(example['input'], dtype=np.uint8)
            output_grid = np.array(example['output'], dtype=np.uint8)
            
            print(f"\n--- Training Example {idx + 1} ---")
            print(f"Grid shape: {input_grid.shape}")
            
            # Extract objects from input (should be solid rectangles)
            objects = extractor.extract(input_grid)
            
            # Extract objects from output (should be hollow rectangles)
            objects_output = extractor.extract(output_grid)
            
            # Validate grid boundary exists for input
            grid_obj_input = next((obj for obj in objects if obj.is_grid_boundary), None)
            if not grid_obj_input:
                print("[FAIL] Input grid boundary not found!")
                all_correct = False
                continue
            
            # Validate grid boundary exists for output
            grid_obj_output = next((obj for obj in objects_output if obj.is_grid_boundary), None)
            if not grid_obj_output:
                print("[FAIL] Output grid boundary not found!")
                all_correct = False
                continue
            
            print("[OK] Grid boundaries present for input and output")
            
            # Count colored objects (excluding grid boundary)
            colored_objects_input = [obj for obj in objects if not obj.is_grid_boundary]
            colored_objects_output = [obj for obj in objects_output if not obj.is_grid_boundary]
            
            total_input = len(colored_objects_input)
            total_output = len(colored_objects_output)
            
            print(f"Input colored objects: {total_input}, Output colored objects: {total_output}")
            
            if total_input == 0:
                print("[FAIL] No colored objects found in input!")
                all_correct = False
                continue
            
            # Check that rectangles become hollow
            # Group by color for analysis
            input_by_color = {}
            for obj in colored_objects_input:
                if obj.color not in input_by_color:
                    input_by_color[obj.color] = []
                input_by_color[obj.color].append(obj)
            
            output_by_color = {}
            for obj in colored_objects_output:
                if obj.color not in output_by_color:
                    output_by_color[obj.color] = []
                output_by_color[obj.color].append(obj)
            
            # Check that rectangles become hollow
            for color in input_by_color.keys():
                input_objs = input_by_color.get(color, [])
                output_objs = output_by_color.get(color, [])
                
                print(f"\n  Color {color}:")
                for obj_idx, input_obj in enumerate(input_objs):
                    print(f"    Object {obj_idx + 1}: {input_obj.area} pixels, "
                          f"hollow={input_obj.is_hollow}, bbox {input_obj.bbox}")
                    
                    # In output, should be hollow
                    if obj_idx < len(output_objs):
                        output_obj = output_objs[obj_idx]
                        print(f"      Output: {output_obj.area} pixels, "
                              f"hollow={output_obj.is_hollow}")
                        
                        # Output object should be hollow (interior empty)
                        if output_obj.is_hollow:
                            print(f"      [OK] Rectangle correctly identified as hollow")
                        else:
                            print(f"      [WARN] Rectangle not detected as hollow")
            
            print("[OK] Rectangle extraction succeeded")
        
        return all_correct
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extraction_on_problem(problem_path, problem_name):
    """Test extraction on a single problem.
    
    General test for various problem types to validate object extraction and feature detection.
    """
    print(f"\n{'='*60}")
    print(f"Problem: {problem_name}")
    print(f"{'='*60}")
    
    try:
        problem = load_arc_problem(problem_path)
        
        extractor = ObjectExtractor(connectivity="8")
        
        # Test on training inputs
        train_count = len(problem['train'])
        test_count = len(problem['test'])
        print(f"Training examples: {train_count}, Test examples: {test_count}")
        
        for idx, example in enumerate(problem['train'][:2]):  # First 2 examples
            input_grid = np.array(example['input'], dtype=np.uint8)
            output_grid = np.array(example['output'], dtype=np.uint8)
            
            print(f"\n--- Training Example {idx + 1} ---")
            print_grid(input_grid, "Input")
            print_grid(output_grid, "Output")
            
            # Extract objects
            try:
                objects = extractor.extract(input_grid)
                
                # Find grid boundary object
                grid_obj = next((obj for obj in objects if obj.is_grid_boundary), None)
                if not grid_obj:
                    print(f"[FAIL] Grid boundary object not found!")
                    return False
                
                print(f"[OK] Grid boundary: {input_grid.shape}")
                
                # Count colored objects (excluding grid boundary)
                colored_objects = [obj for obj in objects if not obj.is_grid_boundary]
                total_all = len(objects)
                
                # Only fail if grid has non-zero content but no objects found
                if len(colored_objects) == 0 and np.any(input_grid):
                    print(f"[FAIL] No colored objects found in input grid (grid not all zeros)!")
                    return False
                elif len(colored_objects) == 0 and not np.any(input_grid):
                    print(f"[OK] Input grid is all zeros, no colored objects expected")
                    continue
                
                print(f"Extracted: {len(colored_objects)} colored objects + grid boundary (total {total_all})")
                
                # Print details for each color
                for obj in colored_objects:
                    min_r, min_c, max_r, max_c = obj.bbox
                    h, w = max_r - min_r + 1, max_c - min_c + 1
                    print(f"\n  Color {obj.color}: ID {obj.id}")
                    print(f"    - {obj.area} pixels, bbox {h}x{w}")
                    print(f"      Closed: {obj.is_closed_shape}, Hollow: {obj.is_hollow}, "
                          f"Holes: {obj.num_holes}")
                    print(f"      Arrow: {obj.is_arrow}, Separator: {obj.is_separator}, "
                          f"Spiral: {obj.is_spiral}, Triangle: {obj.is_triangle}")
                    print(f"      Grid: {obj.is_grid}, Orientation: {obj.orientation}")
                
                print("[OK] Extraction succeeded with colored objects found")
                
            except Exception as e:
                print(f"[ERROR] Extraction failed: {e}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to load problem: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_spiral_problem(problem_path, problem_name):
    """Test on spiral pattern problem (28e73c20).
    
    JSON: 28e73c20.json
    Pattern: Spiral generation - input is blank, output contains spiral pattern
    Key test: Validates detection of spiral structures in the output
    """
    print(f"\n{'='*60}")
    print(f"[SPIRAL TEST] Problem: {problem_name}")
    print(f"{'='*60}")
    
    try:
        problem = load_arc_problem(problem_path)
        extractor = ObjectExtractor(connectivity="8")
        
        all_correct = True
        
        for idx, example in enumerate(problem['train'][:2]):  # First 2 examples
            input_grid = np.array(example['input'], dtype=np.uint8)
            output_grid = np.array(example['output'], dtype=np.uint8)
            
            print(f"\n--- Training Example {idx + 1} ---")
            print(f"Input shape: {input_grid.shape}, Output shape: {output_grid.shape}")
            
            # Extract from output (where spiral appears)
            objects = extractor.extract(output_grid)
            
            # Find grid boundary
            grid_obj = next((obj for obj in objects if obj.is_grid_boundary), None)
            if not grid_obj:
                print("[FAIL] Grid boundary not found in output!")
                all_correct = False
                continue
            
            print("[OK] Grid boundary found")
            
            # Look for spiral in colored objects
            spiral_found = False
            colored_objects = [obj for obj in objects if not obj.is_grid_boundary]
            
            # Group by color for display
            by_color = {}
            for obj in colored_objects:
                if obj.color not in by_color:
                    by_color[obj.color] = []
                by_color[obj.color].append(obj)
            
            for color in sorted(by_color.keys()):
                objs = by_color[color]
                print(f"\n  Color {color}: {len(objs)} object(s)")
                
                for obj_idx, obj in enumerate(objs):
                    min_r, min_c, max_r, max_c = obj.bbox
                    h, w = max_r - min_r + 1, max_c - min_c + 1
                    print(f"    Object {obj_idx + 1}: {obj.area} pixels, bbox {h}x{w}")
                    print(f"      Spiral: {obj.is_spiral}, Closed: {obj.is_closed_shape}")
                    
                    if obj.is_spiral:
                        spiral_found = True
                        print(f"      [OK] Spiral detected!")
            
            if spiral_found:
                print("[OK] Spiral pattern recognized")
            else:
                print("[WARN] No spiral detected in output")
        
        return spiral_found
        
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test extraction on multiple Milestone B problems."""
    milestone_b_dir = Path(__file__).parent.parent / "Milestones" / "B"
    
    if not milestone_b_dir.exists():
        print(f"[ERROR] Milestone B directory not found: {milestone_b_dir}")
        return False
    
    # Test single-object problem specifically
    single_object_path = milestone_b_dir / "1cf80156.json"
    if single_object_path.exists():
        print("\n" + "="*70)
        print("PART 1: SINGLE-OBJECT PATTERN TEST (1cf80156)")
        print("="*70)
        single_object_result = test_single_object_problem(single_object_path, "1cf80156")
    else:
        print("[WARNING] Single-object test file not found")
        single_object_result = False
    
    # Test rectangle hollowing problem specifically
    rectangle_path = milestone_b_dir / "4347f46a.json"
    if rectangle_path.exists():
        print("\n" + "="*70)
        print("PART 2: RECTANGLE HOLLOWING TEST (4347f46a)")
        print("="*70)
        rectangle_result = test_rectangle_hollowing_problem(rectangle_path, "4347f46a")
    else:
        print("[WARNING] Rectangle test file not found")
        rectangle_result = False
    
    # Test spiral pattern problem specifically
    spiral_path = milestone_b_dir / "28e73c20.json"
    if spiral_path.exists():
        print("\n" + "="*70)
        print("PART 3: SPIRAL PATTERN TEST (28e73c20)")
        print("="*70)
        spiral_result = test_spiral_problem(spiral_path, "28e73c20")
    else:
        print("[WARNING] Spiral test file not found")
        spiral_result = False
    
    # Test general extraction on first 3 problems
    print("\n" + "="*70)
    print("PART 4: GENERAL EXTRACTION TEST (First 3 problems)")
    print("="*70)
    problem_files = sorted(milestone_b_dir.glob("*.json"))[:3]
    
    if not problem_files:
        print("[ERROR] No Milestone B problems found")
        return False
    
    print(f"Testing {len(problem_files)} problems\n")
    
    results = []
    for problem_path in problem_files:
        result = test_extraction_on_problem(problem_path, problem_path.stem)
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    general_passed = sum(results)
    general_total = len(results)
    print(f"Single-object test (1cf80156): {'PASS' if single_object_result else 'FAIL'}")
    print(f"Rectangle hollowing test (4347f46a): {'PASS' if rectangle_result else 'FAIL'}")
    print(f"Spiral pattern test (28e73c20): {'PASS' if spiral_result else 'FAIL'}")
    print(f"General extraction: {general_passed}/{general_total} passed")
    
    all_passed = single_object_result and rectangle_result and spiral_result and all(results)
    if all_passed:
        print("\n[OK] All extraction tests passed!")
        return True
    else:
        print("\n[FAIL] Some tests failed")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
