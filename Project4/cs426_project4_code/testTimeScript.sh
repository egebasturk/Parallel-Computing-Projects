echo ---------------TestGroup 1-------------------
echo "\nTesting with cavity02.mtx using 128 Threads, 10 iterations"
./gpummultexec.o 128 10 0 cavity02.mtx

echo "\nTesting with fidapm08.mtx using 128 Threads, 10 iterations"
./gpummultexec.o 128 10 0 fidapm08.mtx

echo "\nTesting with fidapm11.mtx using 128 Threads, 10 iterations"
./gpummultexec.o 128 10 0 fidapm11.mtx

echo "\nTesting with s3dkq4m2.mtx using 128 Threads, 10 iterations"
./gpummultexec.o 128 10 0 s3dkq4m2.mtx

echo "\n---------------TestGroup 2-------------------"
echo "\nTesting with cavity02.mtx using 256 Threads, 10 iterations"
./gpummultexec.o 256 10 0 cavity02.mtx

echo "\nTesting with fidapm08.mtx using 256 Threads, 10 iterations"
./gpummultexec.o 256 10 0 fidapm08.mtx

echo "\nTesting with fidapm11.mtx using 256 Threads, 10 iterations"
./gpummultexec.o 256 10 0 fidapm11.mtx

echo "\nTesting with s3dkq4m2.mtx using 256 Threads, 10 iterations"
./gpummultexec.o 256 10 0 s3dkq4m2.mtx

echo "\n---------------TestGroup 3-------------------"
echo "\nTesting with cavity02.mtx using 512 Threads, 10 iterations"
./gpummultexec.o 512 10 0 cavity02.mtx

echo "\nTesting with fidapm08.mtx using 512 Threads, 10 iterations"
./gpummultexec.o 512 10 0 fidapm08.mtx

echo "\nTesting with fidapm11.mtx using 512 Threads, 10 iterations"
./gpummultexec.o 512 10 0 fidapm11.mtx

echo "\nTesting with s3dkq4m2.mtx using 512 Threads, 10 iterations"
./gpummultexec.o 512 10 0 s3dkq4m2.mtx

echo "\n---------------TestGroup 4-------------------"
echo "\nTesting with cavity02.mtx using 1024 Threads, 10 iterations"
./gpummultexec.o 1024 10 0 cavity02.mtx

echo "\nTesting with fidapm08.mtx using 1024 Threads, 10 iterations"
./gpummultexec.o 1024 10 0 fidapm08.mtx

echo "\nTesting with fidapm11.mtx using 1024 Threads, 10 iterations"
./gpummultexec.o 1024 10 0 fidapm11.mtx

echo "\nTesting with s3dkq4m2.mtx using 1024 Threads, 10 iterations"
./gpummultexec.o 1024 10 0 s3dkq4m2.mtx
