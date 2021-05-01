#include <stdio.h>
#include <pthread.h>
  
void *func(void *arg)
{
    printf("HI from thread %d\n", *(int *)arg);
    return NULL;
}

int main()
{
    int arg[5];
    pthread_t tid[5];
  
    for (int i = 0; i < 5; i++) {
        arg[i] = i;
        pthread_create(tid + i, NULL, func, (void *)(arg + i));
    }
    
    for (int i = 0; i < 5; i++) {
        pthread_join(tid[i], NULL);
        printf("thread %d ends\n", i);
    }
  
    return 0;
}
