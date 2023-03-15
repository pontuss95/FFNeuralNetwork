#include "NNFile.h"

/*double extractMatrixElem(DMatrix Matrix, int row, int col)
{
    return Matrix.p[row + col * Matrix.NRows];
}
void insertMatrixElem(DMatrix *Matrix, double elem, int row, int col)
{
    Matrix->p[row + col * Matrix->NRows] = elem;
}*/

void sighandler(int signum) {
   printf("\nTraining aborted with CTRL+C. Handling of this will be implemented\n");
   exit(1);
}

double Sigmoid(double val)
{
    return 1.0 / (1.0 + exp(-val));
}

double DerivSigmoid(double val)
{
    return exp(-val) / pow(1 + exp(-val), 2.0);
}

double LinearFun(double val)
{
    return val;
}

double derivLinearFun(double val)
{
    return 1;
}

void Layer_forwardProp(Layer *self, double *x, int xLen)
{
    Neuron *currNeur;
    for (int i = 0; i < (self->LSize); i++)
    {
        currNeur = (self->Neurn + i);
        self->output[i] = currNeur->Weights[0]; // Initalize with bias

        for (int j = 0; j < xLen; j++)
        {
            self->output[i] += x[j] * currNeur->Weights[j + 1]; // Propagate input through weights.
        }
        currNeur->derivActFunVal = currNeur->derivActFunc(self->output[i]);
        self->output[i] = currNeur->actFunc(self->output[i]); // Non-liner activation function
    }
}

void forwardProp(Network *self, double *x)
{
    Layer_forwardProp((self->Layers), x, self->InpSize);

    Layer *currLayer;
    for (int i = 1; i < self->NLayers; i++)
    {
        currLayer = (self->Layers + i);
        Layer_forwardProp(currLayer, (currLayer - 1)->output, (currLayer)->LSize);
    }
}

double back_prop(Network *self, double *y, double *x)
{
    double cost = 0;

    Layer *finalLayer = &(self->Layers[self->NLayers - 1]);

    rms(y, finalLayer->output, self->OutSize, &cost, self->DCostDLastLay);

    // First propagate from costfun to output layer.
    Neuron *currNeur = NULL;
    for (int i = 0; i < finalLayer->LSize; i++)
    {
        currNeur = &(finalLayer->Neurn[i]);
        currNeur->derivWrtCostFun = self->DCostDLastLay[i] * currNeur->derivActFunVal;
    }

    // Back-prop to first layer
    Layer *L1;
    Layer *L2;
    for (int i = self->NLayers - 1; i >= 0; i--) // Layer loop
    {
        if (i != 0)
        {
            L1 = &self->Layers[i];
            L2 = &self->Layers[i - 1];
        }
        else
        {
            L1 = &self->Layers[i];
            L2 = NULL;
        }

        for (int j = 0; j < L1->LSize; j++) // Neuron loop
        {
            currNeur = &L1->Neurn[j];

            if (L2 != NULL)
            {
                currNeur->derivWeights[0] = currNeur->derivWrtCostFun;
                for (int q = 0; q < L2->LSize; q++) // PrevNeuronLoop
                {
                    currNeur->derivWeights[q + 1] += currNeur->derivWrtCostFun * L2->output[q];

                    if (j == 0)
                    {
                        L2->Neurn[q].derivWrtCostFun = 0;
                    }
                    L2->Neurn[q].derivWrtCostFun += currNeur->derivWrtCostFun * currNeur->Weights[j + 1]; // bias first weight
                }
            }
            else
            {
                currNeur->derivWeights[0] = currNeur->derivWrtCostFun;
                for (int q = 0; q < self->InpSize; q++) // PrevNeuronLoop
                {
                    currNeur->derivWeights[q + 1] += currNeur->derivWrtCostFun * x[q];
                }
            }
        }
        // After looping through every neuron propagate derivative past the nonlinear activation funf or each previous neuron
        if (L2 != NULL)
        {
            for (int q = 0; q < L2->LSize; q++) // PrevNeuronLoop
            {
                L2->Neurn[q].derivWrtCostFun *= L2->Neurn[q].derivActFunVal;
            }
        }
    }

    return cost;
}

void rms(double *y, double *y_hat, int len, double *rmsVal, double *rmsValDer)
{
    *rmsVal = 0;
    *rmsValDer = 0;
    double cumError = 0;
    int i;
    for (i = 0; i < len; i++)
    {
        *rmsVal += pow(y[i] - y_hat[i], 2);
        cumError += y[i] - y_hat[i];
    }
    *rmsVal /= (double)len;
    *rmsVal = sqrt(*rmsVal);

    if(rmsValDer != NULL){
    for (i = 0; i < len; i++)
    {
        rmsValDer[i] = (-(y[i] - y_hat[i]) * 2) / (*rmsVal * (double)len);
    }
    }
}

void gradientDescent(Network *net, int NDataPoints, double learnRate, double RMSDecay, double momentumDecay)
{
    double stepLength = 0;
    double tmp2 = 0;
    double tmp = 0;
    double eps = 1e-15;
    int prevLaySize = 0;
    for (int i = 0; i < net->NLayers; i++)
    {
        Layer *currLay = &net->Layers[i];
        if (i != 0)
        {
            prevLaySize = (currLay - 1)->LSize;
        }
        else
        {
            prevLaySize = net->InpSize;
        }
        for (int j = 0; j < currLay->LSize; j++)
        {
            Neuron *currNeur = &currLay->Neurn[j];

            for (int q = 0; q < prevLaySize + 1; q++)
            {
                //RMSProp algorithm
                tmp2 = RMSDecay*currNeur->movAvgSquareDer[q] + 
                (1-RMSDecay)*pow(currNeur->derivWeights[q], 2.0);
                
                if(tmp2 != tmp2)
                {
                    printf("NaN\n");
                }
                currNeur->movAvgSquareDer[q] = tmp2;
                tmp = sqrt(currNeur->movAvgSquareDer[q]+eps);
                currNeur->momentumGrad[q] = momentumDecay*currNeur->momentumGrad[q] - learnRate/(tmp)*currNeur->derivWeights[q];
                currNeur->Weights[q] += currNeur->momentumGrad[q];
                currNeur->derivWeights[q] = 0;

                if(currNeur->Weights[q]!=currNeur->Weights[q]){
                    printf("NaN\n");
                }
                
            }
        }
    }
}

void InitializeWeightsAndBiases(Network *Net)
{
    srand((unsigned)time(NULL));
    for (int i = 0; i < Net->NLayers; i++)
    {
        for (int j = 0; j < Net->LSize[i]; j++)
        {
            int iterVar = 0;
            if (i == 0)
            {
                iterVar = Net->InpSize + 1; //+1 due to bias
            }
            else
            {
                iterVar = Net->LSize[i - 1] + 1; //+1 due to bias
            }
            for (int q = 0; q < iterVar; q++)
            {
                Net->Layers[i].Neurn[j].Weights[q] = (double)rand() / (double)RAND_MAX;
                Net->Layers[i].Neurn[j].derivWeights[q] = 0;
                Net->Layers[i].Neurn[j].movAvgSquareDer[q] = 0.01;
                Net->Layers[i].Neurn[j].momentumGrad[q] = 0;

            }
        }
    }
}

void InitializeLayers(Network *Net)
{
    for (int i = 0; i < Net->NLayers; i++)
    {
        Net->Layers[i].Neurn = (Neuron *)malloc(sizeof(Neuron) * Net->LSize[i]);
        Net->Layers[i].output = (double *)malloc(sizeof(double) * Net->LSize[i]);
        // Net->Layers[i].derivWrtCostFun = (double *)malloc(sizeof(double) * Net->LSize[i]);
    }
    for (int i = 0; i < Net->NLayers; i++)
    {
        for (int j = 0; j < Net->LSize[i]; j++)
        {
            if (i == (Net->NLayers - 1))
            {
                Net->Layers[i].Neurn[j].actFunc = &LinearFun;
                Net->Layers[i].Neurn[j].derivActFunc = &derivLinearFun;
            }
            else
            {
                Net->Layers[i].Neurn[j].actFunc = &Sigmoid;
                Net->Layers[i].Neurn[j].derivActFunc = &DerivSigmoid;
            }
            if (i == 0)
            {
                Net->Layers[i].Neurn[j].Weights = (double *)malloc(sizeof(double) * (Net->InpSize + 1));
                Net->Layers[i].Neurn[j].derivWeights = (double *)malloc(sizeof(double) * (Net->InpSize + 1));
                Net->Layers[i].Neurn[j].movAvgSquareDer = (double *)malloc(sizeof(double) * (Net->InpSize + 1));
                Net->Layers[i].Neurn[j].momentumGrad = (double *)malloc(sizeof(double) * (Net->InpSize + 1));
            }
            else
            {
                (Net->Layers[i].Neurn[j].Weights) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
                (Net->Layers[i].Neurn[j].derivWeights) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
                (Net->Layers[i].Neurn[j].movAvgSquareDer) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
                (Net->Layers[i].Neurn[j].momentumGrad) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
            }
        }
    }
}

Network *InitializeNetwork(int NLay, int *Lsize, int inSize, int outSize)
{
    Network *Net = (Network *)malloc(sizeof(Network) * 1);

    Net->InpSize = inSize;
    Net->OutSize = outSize;
    Net->CostFun = &rms;

    Net->Layers = (Layer *)malloc(sizeof(Layer) * NLay);
    Net->NLayers = NLay;
    Net->LSize = (int *)malloc(sizeof(int) * NLay);
    Net->DCostDLastLay = (double *)malloc(sizeof(double) * Net->OutSize);

    for (int i = 0; i < NLay; i++)
    {
        Net->LSize[i] = Lsize[i];
        Net->Layers[i].LSize = Lsize[i]; // RIght now stored at two locations, should probably change this.
    }

    InitializeLayers(Net);
    InitializeWeightsAndBiases(Net);
    return Net;
}

dataSet *GenerateSineInputData(int N)
{

    dataSet *datSet = (dataSet *)malloc(sizeof(dataSet) * 1);

    datSet->yAugmented = (double **)malloc(sizeof(double *) * N);
    datSet->xAugmented = (double **)malloc(sizeof(double *) * N);

    for (int i = 0; i < N; i++)
    {
        datSet->yAugmented[i] = (double *)malloc(sizeof(double) * 2);
        datSet->xAugmented[i] = (double *)malloc(sizeof(double) * 1);
    }

    double PI = 3.14159265;
    for (int i = 0; i < N; i++)
    {
        srand((unsigned)time(NULL));
        datSet->xAugmented[i][0] = ((double)rand() / (double)RAND_MAX) * 2 * PI;
        datSet->yAugmented[i][0] = sin(datSet->xAugmented[i][0]);
        datSet->yAugmented[i][1] = cos(datSet->xAugmented[i][0]);
    }

    return datSet;
}
int checkForNanVals(Network *net)
{
    int prevLaySize = 0;
    int retVal = 0;
    for (int i = 0; i < net->NLayers; i++)
    {
        Layer *currLay = &net->Layers[i];
        if (i != 0)
        {
            prevLaySize = (currLay - 1)->LSize;
        }
        else
        {
            prevLaySize = net->InpSize;
        }
        for (int j = 0; j < currLay->LSize; j++)
        {
            Neuron *currNeur = &currLay->Neurn[j];
            if (currNeur->derivActFunc != currNeur->derivActFunc)
            {
                printf("Nan-detected derivActFun lay %i neuron %i\n", i, j);
                retVal = 1;
            }
            for (int q = 0; q < prevLaySize + 1; q++)
            {
                if (currNeur->Weights[q] != currNeur->Weights[q])
                {
                    printf("Nan-detected weights lay %i neuron %i weight %i \n", i, j, q);
                    retVal = 1;
                }
            }
        }
    }
    return retVal;
}

void printAllVals(Network *Net)
{
    // For debugging purposes
    int nWeights = 0;
    for (int i = 0; i < Net->NLayers; i++)
    { // For every layer
        printf("\n\nVals for layer %i", i);
        if (i == 0)
        {
            nWeights = Net->InpSize + 1;
        }
        else
        {
            nWeights = Net->LSize[i - 1] + 1;
        }
        Layer *currLay = &(Net->Layers[i]);

        for (int j = 0; j < Net->LSize[i]; j++)
        { // For every neyron
            Neuron currNeur = currLay->Neurn[j];
            printf("\n Neuron %i\n", j);
            printf("derivWrtCostFun: %.3f,  derivActFunVal: %.3f, output: %.3f\n", currNeur.derivWrtCostFun, currNeur.derivActFunVal, currLay->output[j]);
            printf("Weights\n");
            for (int q = 0; q < nWeights; q++)
            { // For every weight
                printf("%.4f, ", currNeur.Weights[q]);
            }
            printf("\nderivWeights\n");
            for (int q = 0; q < nWeights; q++)
            { // For every weight
                printf("%.4f, ", currNeur.derivWeights[q]);
            }
        }
    }
}

int main()
{
    signal(SIGINT, sighandler);
    int LSize[3] = {100, 10, 2};
    int NLay = 3;
    int inSize = 1;
    int outSize = 2;

    Network *Net = InitializeNetwork(NLay, LSize, inSize, outSize);


    printAllVals(Net);
    int NData = 100000;
    dataSet *datSet = GenerateSineInputData(NData);
    double RMS = 0;
    double RMSAcc = 0;
    double RMSAccCount = 0;
    int Epochs = 5001;
    int BatchSize = NData;
    int id = 0;
    double LRate = 0;
    srand((unsigned)time(NULL));
    for (int j = 0; j < Epochs; j++)
    {
        for (int i = 0; i < BatchSize; i++)
        {
            id = (int)floor(((double)rand() / (double)RAND_MAX) * NData - 0.000001);
            forwardProp(Net, datSet->xAugmented[id]);
            RMS += back_prop(Net, datSet->yAugmented[id], datSet->xAugmented[id]);

       /*     if (checkForNanVals(Net))
            {
                printf("Found nans inp %i iteration %i\n\n", i, j);
            }*/
        }

        RMS = RMS / (double)BatchSize;
        RMSAcc += RMS;
        RMSAccCount++;
        RMS = 0;
        //double eps = (double)j / Epochs; // From 0 to 1
        //1e-2 to 1e-7
        //double LRate = 0.01 * (1.0 - eps) + eps * 0.0000001;

        double eps = (double)j / Epochs; // From 0 to 1
        LRate = pow(10,-2*(1-eps) -9*(eps));
        if (j % 200 == 0)
        {

            printf("RMS Val %.9f Epoch: %i LRate: %.10f \n", RMSAcc / RMSAccCount, j, LRate);
            RMSAcc = 0;
            RMSAccCount = 0;
        }
        
        gradientDescent(Net, BatchSize, LRate, 0.9, 0.9);
    }

    // printf("\n Outputs \n");
    // for (int i = 0; i < 2; i++)
    //{
    //    printf("%f  ", Net->Layers[1].output[i]);
    //}
    printAllVals(Net);
}