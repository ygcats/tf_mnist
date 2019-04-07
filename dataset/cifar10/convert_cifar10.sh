python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/test_batch.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/test \
    --base_name test

python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/data_batch_1.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/train \
    --base_name train_1

python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/data_batch_2.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/train \
    --base_name train_2

python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/data_batch_3.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/train \
    --base_name train_3

python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/data_batch_4.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/train \
    --base_name train_4

python3 convert_cifar10_png.py \
    --offset 0 \
    --count 10000 \
    --data /home/ygcats/Downloads/cifar-10-batches-bin/data_batch_5.bin \
    --save_dir /home/ygcats/Desktop/tf_mnist/dataset/cifar10/train \
    --base_name train_5

cat train_1.txt train_2.txt train_3.txt train_4.txt train_5.txt > train.txt
rm train_1.txt train_2.txt train_3.txt train_4.txt train_5.txt