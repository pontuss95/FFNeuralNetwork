#include "NNFile.h"

/*double extractMatrixElem(DMatrix Matrix, int row, int col)
{
    return Matrix.p[row + col * Matrix.NRows];
}
void insertMatrixElem(DMatrix *Matrix, double elem, int row, int col)
{
    Matrix->p[row + col * Matrix->NRows] = elem;
}*/

double Sigmoid(double val)
{
    return 1 / (1 + exp(-val));
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
        currNeur = &(self->Layers->Neurn[i]);
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
                    currNeur->derivWeights[q + 1] = currNeur->derivWrtCostFun * L2->output[q];

                    if (j == 0)
                    {
                        L2->Neurn[q].derivWrtCostFun = 0;
                    }
                    L2->Neurn[q].derivWrtCostFun += currNeur->derivWrtCostFun * currNeur->Weights[j + 1];
                }
            }
            else
            {
                currNeur->derivWeights[0] = currNeur->derivWrtCostFun;
                for (int q = 0; q < self->InpSize; q++) // PrevNeuronLoop
                {
                    currNeur->derivWeights[q + 1] = currNeur->derivWrtCostFun * x[q];
                }
            }
        }
    }

    /* for (int i = 0; i < prevToFinalLayer->LSize; i++)
     {
         finalLayer->derivWrtCostFun[i] = 0;
         for (int j = 0; j < finalLayer->LSize; j++)
         {
             CurrNeur = (finalLayer->Neurn + j);
             finalLayer->derivWrtCostFun[i] += CurrNeur->Weights[i] * CurrNeur->derivActFunc(finalLayer->output[j]); //
             CurrNeur->Weights[i] * CurrNeur->Weights[i] * CurrNeur->derivActFunc(finalLayer->output[j]);
         }
     }*/

    return cost;
}

void rms(double *y, double *y_hat, int len, double *rmsVal, double *rmsValDer)
{
    *rmsVal = 0;
    *rmsValDer = 0;
    int i;
    for (i = 0; i < len; i++)
    {
        *rmsVal += pow(y[i], y_hat[i]);
    }
    for (i = 0; i < len; i++)
    {
        rmsValDer[i] = (y_hat[i] - y[i]) / (sqrt((double)len) * sqrt(*rmsVal));
    }
    *rmsVal /= (double)len;
    *rmsVal = sqrt(*rmsVal);
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
                iterVar = Net->InpSize + 1;
            }
            else
            {
                iterVar = Net->LSize[i - 1] + 1;
            }
            for (int q = 0; q < iterVar; q++)
            {
                Net->Layers[i].Neurn[j].Weights[q] = (double)rand() / (double)RAND_MAX;
                Net->Layers[i].Neurn[j].derivWeights[q] = 0;
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
            }
            else
            {
                (Net->Layers[i].Neurn[j].Weights) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
                (Net->Layers[i].Neurn[j].derivWeights) = (double *)malloc(sizeof(double) * (Net->LSize[i - 1] + 1));
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
        datSet->xAugmented[i] = (double *)malloc(sizeof(double) * 2);
    }

    double PI = 3.14159265;
    double maxX, minX, maxY, minY;
    for (int i = 0; i < N; i++)
    {
        srand((unsigned)time(NULL));
        datSet->xAugmented[0][0] = ((double)rand() / (double)RAND_MAX) * 2*PI;
        datSet->xAugmented[0][1] = ((double)rand() / (double)RAND_MAX) * 2*PI;
        datSet->yAugmented[0][0] = sin(datSet->xAugmented[0][0]);
        datSet->yAugmented[0][1] = cos(datSet->xAugmented[0][1]);
        

    }
}

int main()
{
    int LSize[2] = {2, 2};
    int NLay = 2;
    int inSize = 2;
    int outSize = 2;

    Network *Net = InitializeNetwork(NLay, LSize, inSize, outSize);

    double x[2] = {0.12, 0.1421};
    double y[2] = {0.0, 0.0};
    forwardProp(Net, x);

    double RMS = back_prop(Net, y, x);
    printf("\n Outputs \n");
    for (int i = 0; i < 2; i++)
    {
        printf("%f  ", Net->Layers[1].output[i]);
    }
}