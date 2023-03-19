#include "floatCsvReader.h"
#include "FFNeuralNet.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <signal.h>
#include <string.h>


Array *ReadUTFile(char *f){

   char path[200] = "data/";

   strcat(path, f);
   
   FILE *file;
   if(file = fopen(path, "r")){
   printf("Reading file %s", path);
   Array *arr = readCsv(file);

   fclose(file);

   printf("  Read nRows: %d, nCols: %d\n", arr->numRows, arr->numCols);
   return arr;
   }
   else{
      printf("Failed to read file %s filePointer: %p\n", path, file);
      return NULL;
   }
   fflush(stdout);

}

void assertEquality(Layer *L, Array *W, char *testName, double factor){
   double cumSum = 0;
   for(int i = 0; i<W->numRows; i++){
      for(int j = 0; j<W->numCols; j++){
            cumSum += fabs(L->Neurn[i].derivWeights[j] - factor*W->Array[i][j]);
      }
   }

   if(cumSum > 0.00001){
      printf("Test %s FAILED, cumulative error %.10f\n", testName, cumSum);
   }else{
      printf("Test %s PASSED\n", testName);
   }
   
   

}

void assignWeights(Network *Net, Array *W1, Array *W2, Array *W3){
   for(int i = 0; i<W1->numCols; i++){
      for(int j = 0; j<W1->numRows; j++){
         Net->Layers[0].Neurn[j].Weights[i] = W1->Array[j][i];
      }
   }

   for(int i = 0; i<W2->numCols; i++){
      for(int j = 0; j<W2->numRows; j++){
         Net->Layers[1].Neurn[j].Weights[i] = W2->Array[j][i];
      }
   }
   for(int i = 0; i<W3->numCols; i++){
      for(int j = 0; j<W3->numRows; j++){
         Net->Layers[2].Neurn[j].Weights[i] = W3->Array[j][i];
      }
   }

}
int main()
{   

   Array *input = ReadUTFile("inp.csv");
   Array *output = ReadUTFile("out.csv");
   Array *OutLay1 = ReadUTFile("OutLay1.csv");
   Array *OutLay2 = ReadUTFile("OutLay2.csv");
   Array *OutLay3 = ReadUTFile("OutLay3.csv");
   Array *W1 = ReadUTFile("W1.csv");
   Array *W2 = ReadUTFile("W2.csv");
   Array *W3 = ReadUTFile("W3.csv");
   Array *W1Der = ReadUTFile("W1Der.csv");
   Array *W2Der = ReadUTFile("W2Der.csv");
   Array *W3Der = ReadUTFile("W3Der.csv");
   int LSize[3] = {3, 2, 2};
   int NLay = 3;


   Network *Net = InitializeNetwork(NLay, LSize, 2, 2);

   //Setting layers to the same weights as the unit test data


   assignWeights(Net, W1, W2, W3);
   printf("Done assigning weights \n");


   forwardProp(Net, input->Array[0]);
   back_prop(Net, output->Array[0], input->Array[0]);

   assertEquality(&Net->Layers[0], W1Der, "Gradient W1", 1.0);
   assertEquality(&Net->Layers[1], W2Der, "Gradient W2", 1.0);
   assertEquality(&Net->Layers[2], W3Der, "Gradient W3", 1.0);

   forwardProp(Net, input->Array[0]);
   back_prop(Net, output->Array[0], input->Array[0]);

   assertEquality(&Net->Layers[0], W1Der, "Gradient W1 doubleProp", 2.0);
   assertEquality(&Net->Layers[1], W2Der, "Gradient W2 doubleProp", 2.0);
   assertEquality(&Net->Layers[2], W3Der, "Gradient W3 doubleProp", 2.0);
   
   
   
   gradientDescent(Net, 1, 0.01, 0.9, 0.9);
   assignWeights(Net, W1, W2, W3);
   forwardProp(Net, input->Array[0]);
   back_prop(Net, output->Array[0], input->Array[0]);
   assertEquality(&Net->Layers[0], W1Der, "Gradient W1 After backprop", 1.0);
   assertEquality(&Net->Layers[1], W2Der, "Gradient W2 AFter backprop", 1.0);
   assertEquality(&Net->Layers[2], W3Der, "Gradient W3 After backprop", 1.0);

   FILE *fid = fopen("output/W1DerC.csv", "w");
   for(int i = 0; i<W1->numRows; i++){
      for(int j = 0; j<W1->numCols; j++){
         if((j+1) == W1->numCols){
            fprintf(fid, "%.10f\n", Net->Layers[0].Neurn[i].derivWeights[j]);
         }else{

            fprintf(fid, "%.10f, ", Net->Layers[0].Neurn[i].derivWeights[j]);
         }
      }
   }
   fclose(fid);

   fid = fopen("output/W2DerC.csv", "w");
   for(int i = 0; i<W2->numRows; i++){
      for(int j = 0; j<W2->numCols; j++){
         if((j+1) == W2->numCols){
            fprintf(fid, "%.10f\n", Net->Layers[1].Neurn[i].derivWeights[j]);
         }else{

            fprintf(fid, "%.10f, ", Net->Layers[1].Neurn[i].derivWeights[j]);
         }
      }
   }
   fclose(fid);



   fid = fopen("output/W3DerC.csv", "w");
   for(int i = 0; i<W3->numRows; i++){
      for(int j = 0; j<W3->numCols; j++){
         if((j+1) == W3->numCols){
            fprintf(fid, "%.10f\n", Net->Layers[2].Neurn[i].derivWeights[j]);
         }else{

            fprintf(fid, "%.10f, ", Net->Layers[2].Neurn[i].derivWeights[j]);
         }
      }
   }
   fclose(fid);

   fid = fopen("output/RmsDer.csv", "w");
   for(int i = 0; i<2; i++){
      for(int j = 0; j<1; j++){
         if((j+1) == W3->numCols){
            fprintf(fid, "%.10f\n", Net->DCostDLastLay[i]);
         }else{

            fprintf(fid, "%.10f, ", Net->DCostDLastLay[i]);
         }
      }
   }
   fclose(fid);

   fid = fopen("output/CumDerL3C.csv", "w");
   for(int i = 0; i<2; i++){
      for(int j = 0; j<1; j++){
         if((j+1) == W3->numCols){
            fprintf(fid, "%.10f\n", Net->Layers[2].Neurn[i].derivWrtCostFun);
         }else{

            fprintf(fid, "%.10f, ", Net->Layers[2].Neurn[i].derivWrtCostFun);
         }
      }
   }
   fclose(fid);

   fid = fopen("output/CumDerL2C.csv", "w");
   for(int i = 0; i<2; i++){
      for(int j = 0; j<1; j++){
         if((j+1) == W3->numCols){
            fprintf(fid, "%.10f\n", Net->Layers[1].Neurn[i].derivWrtCostFun);
         }else{

            fprintf(fid, "%.10f, ", Net->Layers[1].Neurn[i].derivWrtCostFun);
         }
      }
   }
   fclose(fid);


   return 1;
}