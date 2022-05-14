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

double Layer_forwardProp(Layer self, double *x, int xLen)
{
    Neuron *currNeur;
    for (int i = 0; i < (self.LSize); i++)
    {
        currNeur = (self.Neurn + i);
        *(self.output + i) = *(currNeur->Weight); // Initalize with bias

        for (int j = 1; j < (xLen + 1); j++)
        {
            *(self.output + i) += *(x + j) * *(currNeur->Weight + j); // Propagate input through weights.
        }

        *(self.output + i) = currNeur->actFunc(*(self.output + i)); // Non-liner activation function
    }
}

void forwardProp(Network self, double *x)
{
    Layer_forwardProp(*(self.Layers), x, self.InpSize);

    Layer *currLayer;
    for (int i = 1; i < self.NHiddenLayers; i++)
    {
        currLayer = (self.Layers + i);
        Layer_forwardProp(*currLayer, (*(currLayer - 1)).output, (*currLayer).LSize);
    }
}

double back_prop(Network self, double *y, int yLen, double Norm)
{
    double rmsVal;
    double *rmsDeriv;

    Layer *finalLayer = (self.Layers + self.NHiddenLayers - 1);
    Layer *prevToFinalLayer = (self.Layers + self.NHiddenLayers - 2);

    rms(y, finalLayer->output, yLen, &rmsVal, rmsDeriv);
    int out = 0;
    int inp = 0;
    Neuron *CurrNeur;
    for (int i = 0; i < prevToFinalLayer->LSize; i++)
    {
        *(finalLayer->derivWrtCostFun + i) = 0;
        for (int j = 0; j < finalLayer->LSize; j++)
        {
            CurrNeur = (finalLayer->Neurn + j);
            *(finalLayer->derivWrtCostFun + i) += *(CurrNeur->Weight + i) * CurrNeur->derivActFunc(*(finalLayer->output + j)); //
            *(CurrNeur->Weight + i) * *(CurrNeur->Weight + i) * CurrNeur->derivActFunc(*(finalLayer->output + j));
        }
    }

    return 0;
}

void rms(double *y, double *y_hat, int len, double *rmsVal, double *rmsValDer)
{
    *rmsVal = 0;
    *rmsValDer = 0;
    int i;
    for (i = 0; i < len; i++)
    {
        *rmsVal += pow(*(y + i), *(y_hat + i));
    }
    for (i = 0; i < len; i++)
    {
        *(rmsValDer + i) = (*(y_hat + i) - *(y + i)) / (sqrt((double)len) * sqrt(*rmsVal));
    }
    *rmsVal /= (double)len;
    *rmsVal = sqrt(*rmsVal);
}



void InitializeLayers(Network *Net)
{
    for (int i = 0; i < Net->NLayers; i++)
    {
        Net->Layers->Neurn = (Neuron *)malloc(sizeof(Neuron) * *(Net->LSize + i));
    }
    Net->Layers->Neurn->Weights = (double *)malloc(sizeof(double) * (Net->InpSize+1));
    Net->Layers->Neurn->derivWeights = (double *)malloc(sizeof(double) * (Net->InpSize+1));

    for (int i = 1; i < Net->NLayers ; i++)
    {
    (Net->Layers->Neurn->Weights + i) = (double *)malloc(sizeof(double) * (*(Net->LSize - 1 + i) + 1));
    (Net->Layers->Neurn->derivWeights + i) = (double *)malloc(sizeof(double) * (*(Net->LSize - 1 + i) + 1));


    }
}

Network *InitializeNetwork(int NLay, int *Lsize, int inSize, int outSize)
{

    Network *Net = (Network *)malloc(sizeof(Network) * 1);
    Net->Layers = (Layer *)malloc(sizeof(Layer) * NLay);
    Net->NLayers = NLay;
    Net->LSize = (int *)malloc(sizeof(int) * NLay);

    for (int i = 0; i < NLay; i++)
    {
        *(Net->LSize + i) = *(Lsize + i);
    }

    Net->InpSize = inSize;
    Net->OutSize = outSize;
    Net->CostFun = &rms;
}

int main()
{
}