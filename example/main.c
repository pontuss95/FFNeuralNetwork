#include "floatCsvReader.h"
#include "FFNeuralNet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <signal.h>
#include <string.h>

int main()
{   


    //Reading input and output data
    FILE *fInp = fopen("../inputs.csv", "r");
    FILE *fOut = fopen("../outputs.csv", "r");

    Array *inpArray = readCsv(fInp);
    Array *outArray = readCsv(fOut);
    dataSet *csvDataSet = formatCsvData(inpArray, outArray);
    //dataSet *csvDataSet = generateConstData();
    //dataSet *csvDataSet = generateStep(10);
    //printData(csvDataSet);

    //signal(SIGINT, sighandler);
    int LSize[2] = {1000, 1};
    int NLay = 2;
 

    Network *Net = InitializeNetwork(NLay, LSize, csvDataSet->nInps, csvDataSet->nOuts);

    double gradientSize = 0;

    printAllVals(Net);
    //return 0;
    //int NData = 4000;
    
    double RMS = 0;
    double RMSAcc = 0;
    double RMSAccCount = 0;
    int Epochs = 50000;
    int BatchSize = csvDataSet->nDataPoints;
    int id = 0;
    double LRate = 0.01;
    srand((unsigned)time(NULL));
    for (int j = 0; j < Epochs; j++)
    {
        for (int i = 0; i < BatchSize; i++)
        {
            
            forwardProp(Net, csvDataSet->xAugmented[i]);
            RMS += back_prop(Net, csvDataSet->yAugmented[i], csvDataSet->xAugmented[i]);

       /*     if (checkForNanVals(Net))
            {
                printf("Found nans inp %i iteration %i\n\n", i, j);
            }*/
        }

        //printAllVals(Net);

        RMS = RMS / (double)BatchSize;
        RMSAcc += RMS;
        RMSAccCount++;
        RMS = 0;
        //double eps = (double)j / Epochs; // From 0 to 1
        //1e-2 to 1e-7
        //double LRate = 0.01 * (1.0 - eps) + eps * 0.0000001;

        double eps = (double)j / Epochs; // From 0 to 1
        LRate = pow(10,-2*(1-eps) -6*(eps));
        gradientSize = calculateGradientNorm(Net);
        if (j % 1000 == 0)
        {

            printf("RMS Val %.9f Epoch: %i LRate: %.10f Gradient: %.10f \n", RMSAcc / RMSAccCount, j, LRate, gradientSize);
            RMSAcc = 0;
            RMSAccCount = 0;
        }



        gradientDescent(Net, BatchSize, LRate, 0.9, 0); //First param is RMSPROP and second is Momentum
    }

    // printf("\n Outputs \n");
    // for (int i = 0; i < 2; i++)
    //{
    //    printf("%f  ", Net->Layers[1].output[i]);
    //}
    //printAllVals(Net);
}