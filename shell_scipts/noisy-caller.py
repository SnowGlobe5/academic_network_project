import academic_network_project.anp_core.
import time
import sys
import multiprocessing as mp

mp.set_start_method('spawn', force=True)

def execute_code(arg):
    if arg == '1':
       p1 = 0.5
       p2 = 0.5
       p3 = 0.5
       f = 2

    elif arg == '2':
        p1 = 0.75
        p2 = 0.5
        p3 = 0.5
        f = 2
        
    elif arg == '3':
        p1 = 0.5
        p2 = 0.75
        p3 = 0.5
        f = 2
        
    elif arg == '4':
        p1 = 0.5
        p2 = 0.5
        p3 = 0.75
        f = 2
        
    elif arg == '5':
        p1 = 0.25
        p2 = 0.75
        p3 = 0.25
        f = 2
    else:
        print("Invalid argument. Please provide a valid argument (1, 2, 3, 4, ...).")

    
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [0, 0], 1, 0)))
    p.start()
    time.sleep(60)
    
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [0, 0], 2, 0)))
    p.start()
    time.sleep(60)
    
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [1, 1], None, 0)))
    p.start()
    time.sleep(60)
    
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [2, 3], None, 0)))
    p.start()
    time.sleep(60)
    
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [4, 4], None, 1)))
    p.start()
    time.sleep(60)
        
    p = mp.Process(target=academic_network_project.anp_core.anp_infosphere_expansion_caller.finalize_infosphere, args=(([0, 1, 2, 3, 4], 2019, True, p1, p2, p3, f, [5, 9], None, 1)))
    p.start()
    time.sleep(60)
    
    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python noisy-caller.py <argument>")
    else:
        argument = sys.argv[1]
        execute_code(argument)