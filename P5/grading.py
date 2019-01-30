import glob
import os
import time


def check_grade_pca():
    expected_result_file = 'sol_pca_output.txt'
    result_file = 'pca_output.txt'
    score_count = 0
    if os.path.exists(result_file):
        with open(result_file, 'r') as result:
            with open(expected_result_file, 'r') as expected_result:
                expected_line = expected_result.readline()
                line = result.readline()
                while expected_line and line:
                    try:
                        er = complex(expected_line.strip()).real
                        r = complex(line.strip()).real
                        if abs(er - r) <= 1e-7:
                            score_count += 1
                        expected_line = expected_result.readline()
                        line = result.readline()
                    except:
                        break
    return float(score_count) / 42000.0 * 2.5


def check_grade_hmm():
    log_prob_solution = -8.15031
    score = 0
    try:
        filename = glob.glob("hmm.txt")
        student = open(filename[0], "r")
        ans = student.readlines()
        log_prob = float(ans[0].split(",")[1].strip()[:-1])
        if abs(log_prob_solution - log_prob) <= 1e-4:
            score += 0.7
        path_values = ans[2].strip().split(" ")
        for p in path_values:
            if int(p) == 1:
                score += 0.3
        score = min(score, 2.5)
    except:
        pass
    return score


def run_hmm_code():
    try:
        os.system("python3 hmm.py hmm_model.json AGCGTA | tee hmm.txt")
        return True
    except Exception as e:
        print(e)
    return False


def run_pca_code():
    try:
        start_time = time.time()
        os.system("python3 pca.py")
        time_taken = time.time() - start_time
        if time_taken <= 60:
            return True
    except Exception as e:
        print(e)
        return False
    print("Runtime < 1 min constraint not satisfied")
    return False


if __name__ == "__main__":

    score_hmm = 0.0
    if run_hmm_code():
        score_hmm = check_grade_hmm()

    score_pca = 0.0
    if run_pca_code():
        score_pca = check_grade_pca()

    score = 0.0
    score += score_hmm
    score += score_pca

    with open('output_hw5.txt', 'w') as f:
        f.write("[HMM score]:" + str(score_hmm) + '\n')
        f.write("[PCA score]:" + str(score_pca) + '\n')
        f.write("[Total]:" + str(score))