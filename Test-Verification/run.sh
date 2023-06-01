CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type ArcFace-r100 --real-dir ./results/results_test1/real_ima --fake-dir ./results/results_test1/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type CosFace-r100 --real-dir ./results/results_test1/real_ima --fake-dir ./results/results_test1/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type VGGFace2 --real-dir ./results/results_test1/real_ima --fake-dir ./results/results_test1/fake_ima

CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type ArcFace-r100 --real-dir ./results/results_test2/real_ima --fake-dir ./results/results_test2/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type CosFace-r100 --real-dir ./results/results_test2/real_ima --fake-dir ./results/results_test2/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type VGGFace2 --real-dir ./results/results_test2/real_ima --fake-dir ./results/results_test2/fake_ima

CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type ArcFace-r100 --real-dir ./results/results_test3/real_ima --fake-dir ./results/results_test3/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type CosFace-r100 --real-dir ./results/results_test3/real_ima --fake-dir ./results/results_test3/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type VGGFace2 --real-dir ./results/results_test3/real_ima --fake-dir ./results/results_test3/fake_ima

CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type ArcFace-r100 --real-dir ./results/results_test4/real_ima --fake-dir ./results/results_test4/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type CosFace-r100 --real-dir ./results/results_test4/real_ima --fake-dir ./results/results_test4/fake_ima
CUDA_VISIBLE_DEVICES=1 python Verification.py --Net-type VGGFace2 --real-dir ./results/results_test4/real_ima --fake-dir ./results/results_test4/fake_ima
