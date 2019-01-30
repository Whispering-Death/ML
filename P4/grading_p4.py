import numpy as np
import argparse
import os
import time


def test_kmeans_toy(results):
    score = 0

    try:
        f_name = 'k_means_toy.npz'
        data = np.load('{}/{}'.format(results, f_name))
        centroids, step, membership, y = data['centroids'], data['step'], data[
            'membership'], data['y']

        # compute permutation y:membership
        perm, correct = {}, 0
        for i in range(4):
            temp = membership[y == i]
            unique, counts = np.unique(temp, return_counts=True)
            perm[i] = unique[np.argmax(counts)]
            correct += np.max(counts)

        # check if perm has all unique value
        unique_perm = len(np.unique(list(perm.values()))) == 4
        k = 197  # model solution
        if unique_perm and correct >= k:
            score += 3
        elif unique_perm and correct >= k * 0.8:
            score += 1

        # 3 ideal
        if step <= 3:
            score += 3
        elif step == 4:
            score += 1

    except Exception as e:
        print(e)
    finally:
        return score


def test_kmeans_compression(results):
    score = 0
    try:
        f_name = 'k_means_compression.npz'
        data = np.load('{}/{}'.format(results, f_name))
        centroids, step, new_im, pixel_error, im = data['centroids'], data[
            'step'], data['new_image'], data['pixel_error'], data['im']

        model_pixel_error = 0.009735123193995537
        rounded_model_pixel_error = round(float(model_pixel_error), 6)
        rounded_student_pixel_error = round(float(pixel_error), 6)
        print(pixel_error)
        if rounded_student_pixel_error <= rounded_model_pixel_error:
            score += 8
        elif rounded_student_pixel_error <= rounded_model_pixel_error * 1.01:
            score += 6
        elif rounded_student_pixel_error <= rounded_model_pixel_error * 1.1:
            score += 4

    except Exception as e:
        print(e)
    finally:
        return score


def test_kmeans_classification(results):
    score = 0
    try:
        f_name = 'k_means_classification.npz'
        data = np.load('{}/{}'.format(results, f_name))
        y_hat_test, y_test, centroids, centroid_labels = data[
                                                             'y_hat_test'], data['y_test'], data['centroids'], data[
                                                             'centroid_labels']

        acc = np.mean(y_hat_test == y_test)
        acc = round(float(acc), 6)
        model_accuracy = 0.9933333333333333
        model_accuracy = round(float(model_accuracy), 6)
        if acc >= model_accuracy:
            score += 4
        elif acc >= .98 * model_accuracy:
            score += 2

    except Exception as e:
        print(e)
    finally:
        return score


def __test_gmm_results(data, string, ideal_iterations, ideal_log_l):
    iterations, variances, pi_k, means, log_likelihood, x, y = data[
                                                                   'iterations'], data['variances'], data['pi_k'], data[
                                                                   'means'], data[
                                                                   'log_likelihood'], data['x'], data['y']

    score = 0

    log_likelihood = round(float(log_likelihood), 4)
    ideal_log_l = round(float(ideal_log_l), 4)
    iterations = int(iterations)

    if iterations <= ideal_iterations:
        score += 3
    elif iterations - ideal_iterations == 1:
        score += 1

    if ideal_log_l <= log_likelihood:
        score += 4
    elif np.abs(log_likelihood - ideal_log_l) < 0.2 * np.abs(ideal_log_l):
        score += 2

    return score


def test_gmm_toy(results):
    score = 0

    params = [{
        'fname': 'gmm_toy_k_means.npz',
        'ideal_iterations': 9,
        'ideal_log_l': -1663.2697448261301
    }, {
        'fname': 'gmm_toy_random.npz',
        'ideal_iterations': 30,
        'ideal_log_l': -1663.2697447807122
    }]

    for param in params:
        try:
            data = np.load('{}/{}'.format(results, param['fname']))
            s = __test_gmm_results(data, 'toy', param['ideal_iterations'],
                                   param['ideal_log_l'])
            score += s
        except Exception as e:
            print(e)
    return score


def test_gmm_digits(results):
    params = [{
        'fname': 'gmm_digits_k_means.npz',
        'ideal_iterations': 15,
        'ideal_log_l': 126125.97492355896
    }, {
        'fname': 'gmm_digits_random.npz',
        'ideal_iterations': 10,
        'ideal_log_l': 120308.44272256458
    }]

    score = 0

    for param in params:
        try:
            data = np.load('{}/{}'.format(results, param['fname']))
            s = __test_gmm_results(data, 'digits', param['ideal_iterations'],
                                   param['ideal_log_l'])
            score += s
        except Exception as e:
            print(e)
    return score


def run_student_code():
    score = 0
    try:
        start_time = time.time()
        os.system("python3 kmeansTest.py")
        os.system("python3 gmmTest.py")
        time_taken = time.time() - start_time
        if time_taken <= 50:
            score += 4
    except Exception as e:
        print(e)
        return 0
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grading script for PA4')
    parser.add_argument('-r', help="assignment folder", default="results/")
    args = parser.parse_args()

    score = 0
    #
    # running_time_score = run_student_code()
    # score += running_time_score

    k_means_toy_score = test_kmeans_toy(args.r)
    score += k_means_toy_score

    k_means_compression_score = test_kmeans_compression(args.r)
    score += k_means_compression_score

    k_means_classification_score = test_kmeans_classification(args.r)
    score += k_means_classification_score

    gmm_toys_score = test_gmm_toy(args.r)
    score += gmm_toys_score

    gmm_digits_score = test_gmm_digits(args.r)
    score += gmm_digits_score

    fout = open("output_hw4.txt", 'w')
    # fout.write("[running_time_score]" + ":" + str(running_time_score) + "\n")
    fout.write("[test_k_means_toy]" + ":" + str(k_means_toy_score) + "\n")
    fout.write("[test_k_means_compression_score]" + ":" + str(k_means_compression_score) + "\n")
    fout.write("[test_k_means_classification_score]" + ":" + str(k_means_classification_score) + "\n")
    fout.write("[test_gmm_toys_score]" + ":" + str(gmm_toys_score) + "\n")
    fout.write("[test_gmm_digits_score]" + ":" + str(gmm_digits_score) + "\n")
    fout.write("[total]" + ":" + str(score) + "\n")
    fout.close()