import time
import numpy as np
from utils.generateSamples import genSample_SSKP
from ParallelSolve import majority_vote, baggingTwoPhase_woSplit, baggingTwoPhase_wSplit, gurobi_matching, gurobi_matching_DRO_wasserstein

# generate weight matrix for the matching problem
def generateW(N, option = None):
    # generate the weight matrix for the maximum weight bipartite matching problem
    # using 0-based index, need N >= 6
    w = {}
    for i in range(N):
        for j in range(N):
            if i < 3 and j < 3:
                w[(i,j)] = None
            elif i >= N-2 and j >= N-2:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    # w[(i,j)] = 2
                    w[(i,j)] = np.random.uniform(1.9, 2.1)
            else:
                if option == "random":
                    w[(i,j)] = np.random.uniform(1.95, 2.05)
                else:
                    # w[(i,j)] = np.random.uniform(1.9, 2)
                    w[(i,j)] = 2.1
    return w

def matching_obj_optimal(sample_args, N, w):
    # computes the optimal objective value (no randomness)
    if sample_args['type'] == 'pareto' or sample_args['type'] == 'sym_pareto' or sample_args['type'] == 'neg_pareto':
        sample_mean = np.reshape([item/(item-1) for item in sample_args['params']], (1, len(sample_args['params'])))
    elif sample_args['type'] == 'normal':
        sample_mean = np.reshape(sample_args['params'][0], (1, len(sample_args['params'][0])))
    x_opt = gurobi_matching(sample_mean, N, w)
    obj = matching_evaluate_exact(sample_args, x_opt, N, w)
    return obj, x_opt
    

def matching_evaluate_exact(sample_args, x, N, w):
    # evaluate the objective value of a given solution (no randomness)
    # x is the solution, represented as a tuple
    # first, retrieve the sample mean and fill the None values in w
    if sample_args['type'] == 'pareto' or sample_args['type'] == 'sym_pareto' or sample_args['type'] == 'neg_pareto':
        sample_mean = [item/(item-1) for item in sample_args['params']]
    elif sample_args['type'] == 'normal':
        sample_mean = sample_args['params'][0]
    
    ind = 0
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    # note that, indices of x correspond to (0,0), (0,2), ..., (0,N-1), (1,0), (1,1), ..., (N-1,N-1)
    edges = [(i, j) for i in range(N) for j in range(N)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1

    return obj

def matching_evaluate_wSol(sample_k, x, N, w):
    # the sample-based version of matching_evaluate_exact function, also returns the solution
    ind = 0
    sample_mean = np.mean(sample_k, axis=0)
    for i in range(3):
        for j in range(3):
            w[(i,j)] = sample_mean[ind]
            ind += 1
    
    edges = [(i, j) for i in range(N) for j in range(N)]
    ind, obj = 0, 0
    for edge in edges:
        obj += w[edge] * x[ind]
        ind += 1

    return obj, x

def comparison_twoPhase(B_list, k_list, B12_list, epsilon, tolerance, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    # for maximum weight matching problem, prob_args represent N and w.
    SAA_list = []
    bagging_alg1_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        bagging_alg1_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind2, (num,ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}")
        SAA_list.append(SAA_intermediate)
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_alg1_list[ind1][ind2].append(bagging_alg1_intermediate[ind1][ind2])
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
    
    return SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list

def evaluation_twoPhase(SAA_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    B_list_len, k_list_len = len(bagging_alg1_list), len(bagging_alg1_list[0])
    B12_list_len = len(bagging_alg3_list)
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind1 in range(B_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg1_list[ind1][ind2][i][j])
            for ind1 in range(B12_list_len):
                for ind2 in range(k_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])

    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
    SAA_obj_list, SAA_obj_avg = [], []
    bagging_alg1_obj_list, bagging_alg1_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind1 in range(B_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list = []
                for j in range(number_of_iterations):
                    bagging_obj = solution_obj_values[str(bagging_alg1_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_alg1_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_alg1_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
            
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = solution_obj_values[str(bagging_alg3_list[ind1][ind2][i][j])]
                    bagging_obj_4 = solution_obj_values[str(bagging_alg4_list[ind1][ind2][i][j])]
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg


def comparison_DRO(B_list, k_list, B12_list, epsilon, tolerance, varepsilon_list, number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    SAA_list = []
    dro_wasserstein_list = [[] for _ in range(len(varepsilon_list))]
    bagging_alg1_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
    bagging_alg3_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
    bagging_alg4_list = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]

    for n in sample_number:
        SAA_intermediate = []
        dro_wasserstein_intermediate = [[] for _ in range(len(varepsilon_list))]
        bagging_alg1_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B_list))]
        bagging_alg3_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        bagging_alg4_intermediate = [[ [] for _ in range(len(k_list)) ] for _ in range(len(B12_list))]
        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            for ind, varepsilon in enumerate(varepsilon_list):
                tic = time.time()
                dro_wasserstein = gurobi_matching_DRO_wasserstein(sample_n, *prob_args, varepsilon= varepsilon)
                dro_wasserstein = tuple([round(x) for x in dro_wasserstein])
                dro_wasserstein_intermediate[ind].append(dro_wasserstein)
                print(f"Sample size {n}, iteration {iter}, varepsilon={varepsilon}, DRO time: {time.time()-tic}")
            
            for ind2, (num, ratio) in enumerate(k_list):
                k = max(num, int(n*ratio))
                for ind1, B in enumerate(B_list):
                    tic = time.time()
                    bagging, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
                    bagging_alg1_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging]))
                    print(f"Sample size {n}, iteration {iter}, B={B}, k={k}, Bagging Alg 1 time: {time.time()-tic}")

                for ind1, (B1, B2) in enumerate(B12_list):
                    tic = time.time()
                    bagging_alg3, _, _, _ = baggingTwoPhase_woSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg4, _, _, _ = baggingTwoPhase_wSplit(sample_n, B1, B2, k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
                    bagging_alg3_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg3]))
                    bagging_alg4_intermediate[ind1][ind2].append(tuple([round(x) for x in bagging_alg4]))
                    print(f"Sample size {n}, iteration {iter}, B1={B1}, B2={B2}, k={k}, Bagging Alg 3 & 4time: {time.time()-tic}")
        
        SAA_list.append(SAA_intermediate)

        for ind in range(len(varepsilon_list)):
            dro_wasserstein_list[ind].append(dro_wasserstein_intermediate[ind])
        
        for ind1 in range(len(B_list)):
            for ind2 in range(len(k_list)):
                bagging_alg1_list[ind1][ind2].append(bagging_alg1_intermediate[ind1][ind2])
        
        for ind1 in range(len(B12_list)):
            for ind2 in range(len(k_list)):
                bagging_alg3_list[ind1][ind2].append(bagging_alg3_intermediate[ind1][ind2])
                bagging_alg4_list[ind1][ind2].append(bagging_alg4_intermediate[ind1][ind2])
        
    return SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list

def evaluation_DRO(SAA_list, dro_wasserstein_list, bagging_alg1_list, bagging_alg3_list, bagging_alg4_list, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    varepsilon_list_len = len(dro_wasserstein_list)
    B_list_len, k_list_len = len(bagging_alg1_list), len(bagging_alg1_list[0])
    B12_list_len = len(bagging_alg3_list)
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            for ind in range(varepsilon_list_len):
                all_solutions.add(dro_wasserstein_list[ind][i][j])
            for ind2 in range(k_list_len):
                for ind1 in range(B_list_len):
                    all_solutions.add(bagging_alg1_list[ind1][ind2][i][j])
                for ind1 in range(B12_list_len):
                    all_solutions.add(bagging_alg3_list[ind1][ind2][i][j])
                    all_solutions.add(bagging_alg4_list[ind1][ind2][i][j])

    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
    SAA_obj_list, SAA_obj_avg = [], []
    dro_wasserstein_obj_list, dro_wasserstein_obj_avg = [[] for _ in range(varepsilon_list_len)], [[] for _ in range(varepsilon_list_len)]
    bagging_alg1_obj_list, bagging_alg1_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B_list_len)]
    bagging_alg3_obj_list, bagging_alg3_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]
    bagging_alg4_obj_list, bagging_alg4_obj_avg = [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)], [[ [] for _ in range(k_list_len) ] for _ in range(B12_list_len)]

    for i in range(sample_number_len):
        current_SAA_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))

    for ind in range(varepsilon_list_len):
        for i in range(sample_number_len):
            current_dro_wasserstein_obj_list = []
            for j in range(number_of_iterations):
                dro_wasserstein_obj = solution_obj_values[str(dro_wasserstein_list[ind][i][j])]
                current_dro_wasserstein_obj_list.append(dro_wasserstein_obj)
            dro_wasserstein_obj_list[ind].append(current_dro_wasserstein_obj_list)
            dro_wasserstein_obj_avg[ind].append(np.mean(current_dro_wasserstein_obj_list))
    
    for ind1 in range(B_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list = []
                for j in range(number_of_iterations):
                    bagging_obj = solution_obj_values[str(bagging_alg1_list[ind1][ind2][i][j])]
                    current_bagging_obj_list.append(bagging_obj)
                bagging_alg1_obj_list[ind1][ind2].append(current_bagging_obj_list)
                bagging_alg1_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list))
            
    for ind1 in range(B12_list_len):
        for ind2 in range(k_list_len):
            for i in range(sample_number_len):
                current_bagging_obj_list_3 = []
                current_bagging_obj_list_4 = []
                for j in range(number_of_iterations):
                    bagging_obj_3 = solution_obj_values[str(bagging_alg3_list[ind1][ind2][i][j])]
                    bagging_obj_4 = solution_obj_values[str(bagging_alg4_list[ind1][ind2][i][j])]
                    current_bagging_obj_list_3.append(bagging_obj_3)
                    current_bagging_obj_list_4.append(bagging_obj_4)
                bagging_alg3_obj_list[ind1][ind2].append(current_bagging_obj_list_3)
                bagging_alg3_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_3))
                bagging_alg4_obj_list[ind1][ind2].append(current_bagging_obj_list_4)
                bagging_alg4_obj_avg[ind1][ind2].append(np.mean(current_bagging_obj_list_4))

    return SAA_obj_list, SAA_obj_avg, dro_wasserstein_obj_list, dro_wasserstein_obj_avg, bagging_alg1_obj_list, bagging_alg1_obj_avg, bagging_alg3_obj_list, bagging_alg3_obj_avg, bagging_alg4_obj_list, bagging_alg4_obj_avg


def comparison_many_methods(B,k_tuple,B12,epsilon,tolerance,varepsilon,number_of_iterations,sample_number,rng_sample, rng_alg, sample_args, *prob_args):
    SAA_list = []
    dro_wasserstein_list = []
    bagging_alg1_SAA_list = []
    bagging_alg1_DRO_list = []
    bagging_alg3_SAA_list = []
    bagging_alg3_DRO_list = []
    bagging_alg4_SAA_list = []
    bagging_alg4_DRO_list = []

    num, ratio = k_tuple
    for n in sample_number:
        k = max(num, int(n*ratio))
        SAA_intermediate = []
        dro_wasserstein_intermediate = []
        bagging_alg1_SAA_intermediate = []
        bagging_alg1_DRO_intermediate = []
        bagging_alg3_SAA_intermediate = []
        bagging_alg3_DRO_intermediate = []
        bagging_alg4_SAA_intermediate = []
        bagging_alg4_DRO_intermediate = []

        for iter in range(number_of_iterations):
            tic = time.time()
            sample_n = genSample_SSKP(n, rng_sample, type = sample_args['type'], params = sample_args['params'])
            SAA, _ = majority_vote(sample_n, 1, n, gurobi_matching, rng_alg, *prob_args)
            SAA_intermediate.append(tuple([round(x) for x in SAA]))
            print(f"Sample size {n}, iteration {iter}, SAA time: {time.time()-tic}")

            tic = time.time()
            dro_wasserstein = gurobi_matching_DRO_wasserstein(sample_n, *prob_args, varepsilon= varepsilon)
            dro_wasserstein = tuple([round(x) for x in dro_wasserstein])
            dro_wasserstein_intermediate.append(dro_wasserstein)
            print(f"Sample size {n}, iteration {iter}, DRO time: {time.time()-tic}")

            tic = time.time()
            bagging_alg1_SAA, _ = majority_vote(sample_n, B, k, gurobi_matching, rng_alg, *prob_args)
            bagging_alg1_DRO, _ = majority_vote(sample_n, B, k, gurobi_matching_DRO_wasserstein, rng_alg, *prob_args, varepsilon =varepsilon)

            bagging_alg3_SAA, _, _, _ = baggingTwoPhase_woSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
            bagging_alg3_DRO, _, _, _ = baggingTwoPhase_woSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching_DRO_wasserstein, matching_evaluate_wSol, rng_alg, *prob_args, varepsilon = varepsilon)

            bagging_alg4_SAA, _, _, _ = baggingTwoPhase_wSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching, matching_evaluate_wSol, rng_alg, *prob_args)
            bagging_alg4_DRO, _, _, _ = baggingTwoPhase_wSplit(sample_n, B12[0], B12[1], k, epsilon, tolerance, gurobi_matching_DRO_wasserstein, matching_evaluate_wSol, rng_alg, *prob_args, varepsilon = varepsilon)

            bagging_alg1_SAA_intermediate.append(tuple([round(x) for x in bagging_alg1_SAA]))
            bagging_alg1_DRO_intermediate.append(tuple([round(x) for x in bagging_alg1_DRO]))
            bagging_alg3_SAA_intermediate.append(tuple([round(x) for x in bagging_alg3_SAA]))
            bagging_alg3_DRO_intermediate.append(tuple([round(x) for x in bagging_alg3_DRO]))
            bagging_alg4_SAA_intermediate.append(tuple([round(x) for x in bagging_alg4_SAA]))
            bagging_alg4_DRO_intermediate.append(tuple([round(x) for x in bagging_alg4_DRO]))
            print(f"Sample size {n}, iteration {iter}, Bagging time: {time.time()-tic}")

        SAA_list.append(SAA_intermediate)
        dro_wasserstein_list.append(dro_wasserstein_intermediate)
        bagging_alg1_SAA_list.append(bagging_alg1_SAA_intermediate)
        bagging_alg1_DRO_list.append(bagging_alg1_DRO_intermediate)
        bagging_alg3_SAA_list.append(bagging_alg3_SAA_intermediate)
        bagging_alg3_DRO_list.append(bagging_alg3_DRO_intermediate)
        bagging_alg4_SAA_list.append(bagging_alg4_SAA_intermediate)
        bagging_alg4_DRO_list.append(bagging_alg4_DRO_intermediate)

    return SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list

def evaluation_many_methods(SAA_list, dro_wasserstein_list, bagging_alg1_SAA_list, bagging_alg1_DRO_list, bagging_alg3_SAA_list, bagging_alg3_DRO_list, bagging_alg4_SAA_list, bagging_alg4_DRO_list, sample_args, *prob_args):
    sample_number_len = len(SAA_list)
    number_of_iterations = len(SAA_list[0])
    all_solutions = set()
    for i in range(sample_number_len):
        for j in range(number_of_iterations):
            all_solutions.add(SAA_list[i][j])
            all_solutions.add(dro_wasserstein_list[i][j])
            all_solutions.add(bagging_alg1_SAA_list[i][j])
            all_solutions.add(bagging_alg1_DRO_list[i][j])
            all_solutions.add(bagging_alg3_SAA_list[i][j])
            all_solutions.add(bagging_alg3_DRO_list[i][j])
            all_solutions.add(bagging_alg4_SAA_list[i][j])
            all_solutions.add(bagging_alg4_DRO_list[i][j])
    
    solution_obj_values = {}
    for solution in all_solutions:
        solution_obj_values[str(solution)] = matching_evaluate_exact(sample_args, solution, *prob_args)
    
    SAA_obj_list, SAA_obj_avg = [], []
    dro_wasserstein_obj_list, dro_wasserstein_obj_avg = [], []
    bagging_alg1_SAA_obj_list, bagging_alg1_SAA_obj_avg = [], []
    bagging_alg1_DRO_obj_list, bagging_alg1_DRO_obj_avg = [], []
    bagging_alg3_SAA_obj_list, bagging_alg3_SAA_obj_avg = [], []
    bagging_alg3_DRO_obj_list, bagging_alg3_DRO_obj_avg = [], []
    bagging_alg4_SAA_obj_list, bagging_alg4_SAA_obj_avg = [], []
    bagging_alg4_DRO_obj_list, bagging_alg4_DRO_obj_avg = [], []

    for i in range(sample_number_len):
        current_SAA_obj_list = []
        current_dro_wasserstein_obj_list = []
        current_bagging_alg1_SAA_obj_list = []
        current_bagging_alg1_DRO_obj_list = []
        current_bagging_alg3_SAA_obj_list = []
        current_bagging_alg3_DRO_obj_list = []
        current_bagging_alg4_SAA_obj_list = []
        current_bagging_alg4_DRO_obj_list = []
        for j in range(number_of_iterations):
            SAA_obj = solution_obj_values[str(SAA_list[i][j])]
            dro_wasserstein_obj = solution_obj_values[str(dro_wasserstein_list[i][j])]
            bagging_alg1_SAA_obj = solution_obj_values[str(bagging_alg1_SAA_list[i][j])]
            bagging_alg1_DRO_obj = solution_obj_values[str(bagging_alg1_DRO_list[i][j])]
            bagging_alg3_SAA_obj = solution_obj_values[str(bagging_alg3_SAA_list[i][j])]
            bagging_alg3_DRO_obj = solution_obj_values[str(bagging_alg3_DRO_list[i][j])]
            bagging_alg4_SAA_obj = solution_obj_values[str(bagging_alg4_SAA_list[i][j])]
            bagging_alg4_DRO_obj = solution_obj_values[str(bagging_alg4_DRO_list[i][j])]
            current_SAA_obj_list.append(SAA_obj)
            current_dro_wasserstein_obj_list.append(dro_wasserstein_obj)
            current_bagging_alg1_SAA_obj_list.append(bagging_alg1_SAA_obj)
            current_bagging_alg1_DRO_obj_list.append(bagging_alg1_DRO_obj)
            current_bagging_alg3_SAA_obj_list.append(bagging_alg3_SAA_obj)
            current_bagging_alg3_DRO_obj_list.append(bagging_alg3_DRO_obj)
            current_bagging_alg4_SAA_obj_list.append(bagging_alg4_SAA_obj)
            current_bagging_alg4_DRO_obj_list.append(bagging_alg4_DRO_obj)
        SAA_obj_list.append(current_SAA_obj_list)
        dro_wasserstein_obj_list.append(current_dro_wasserstein_obj_list)
        bagging_alg1_SAA_obj_list.append(current_bagging_alg1_SAA_obj_list)
        bagging_alg1_DRO_obj_list.append(current_bagging_alg1_DRO_obj_list)
        bagging_alg3_SAA_obj_list.append(current_bagging_alg3_SAA_obj_list)
        bagging_alg3_DRO_obj_list.append(current_bagging_alg3_DRO_obj_list)
        bagging_alg4_SAA_obj_list.append(current_bagging_alg4_SAA_obj_list)
        bagging_alg4_DRO_obj_list.append(current_bagging_alg4_DRO_obj_list)
        SAA_obj_avg.append(np.mean(current_SAA_obj_list))
        dro_wasserstein_obj_avg.append(np.mean(current_dro_wasserstein_obj_list))
        bagging_alg1_SAA_obj_avg.append(np.mean(current_bagging_alg1_SAA_obj_list))
        bagging_alg1_DRO_obj_avg.append(np.mean(current_bagging_alg1_DRO_obj_list))
        bagging_alg3_SAA_obj_avg.append(np.mean(current_bagging_alg3_SAA_obj_list))
        bagging_alg3_DRO_obj_avg.append(np.mean(current_bagging_alg3_DRO_obj_list))
        bagging_alg4_SAA_obj_avg.append(np.mean(current_bagging_alg4_SAA_obj_list))
        bagging_alg4_DRO_obj_avg.append(np.mean(current_bagging_alg4_DRO_obj_list))

        evaluation_results = {
            "SAA_obj_list": SAA_obj_list,
            "SAA_obj_avg": SAA_obj_avg,
            "dro_wasserstein_obj_list": dro_wasserstein_obj_list,
            "dro_wasserstein_obj_avg": dro_wasserstein_obj_avg,
            "bagging_alg1_SAA_obj_list": bagging_alg1_SAA_obj_list,
            "bagging_alg1_SAA_obj_avg": bagging_alg1_SAA_obj_avg,
            "bagging_alg1_DRO_obj_list": bagging_alg1_DRO_obj_list,
            "bagging_alg1_DRO_obj_avg": bagging_alg1_DRO_obj_avg,
            "bagging_alg3_SAA_obj_list": bagging_alg3_SAA_obj_list,
            "bagging_alg3_SAA_obj_avg": bagging_alg3_SAA_obj_avg,
            "bagging_alg3_DRO_obj_list": bagging_alg3_DRO_obj_list,
            "bagging_alg3_DRO_obj_avg": bagging_alg3_DRO_obj_avg,
            "bagging_alg4_SAA_obj_list": bagging_alg4_SAA_obj_list,
            "bagging_alg4_SAA_obj_avg": bagging_alg4_SAA_obj_avg,
            "bagging_alg4_DRO_obj_list": bagging_alg4_DRO_obj_list,
            "bagging_alg4_DRO_obj_avg": bagging_alg4_DRO_obj_avg
        }

    return evaluation_results