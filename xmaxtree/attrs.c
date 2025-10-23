/* attrs.c */
/* October 2000  Erik Urbach */
#include <math.h>
#include <stdlib.h>
#include "attrs.h"
#include "maxtree.h"
#include "mtmacros.h"
#include "mtconstants.h"
#include "mttypes.h"



/***TODO: remove these lines ***/
extern long ImageWidth;
extern ubyte *ORI;
/*******************************/



/****** Typedefs and functions for area attributes ******************************/

typedef struct AreaData
{
   long Area;
} AreaData;

void *NewAreaData(int x, int y)
{
   AreaData *areadata;

   areadata = malloc(sizeof(AreaData));
   areadata->Area = 1;
   return(areadata);
} /* NewAreaData */

void DeleteAreaData(void *areaattr)
{
   free(areaattr);
} /* DeleteAreaData */

void AddToAreaData(void *areaattr, int x, int y)
{
   AreaData *areadata = areaattr;

   areadata->Area ++;
} /* AddToAreaData */

void MergeAreaData(void *areaattr, void *childattr)
{
   AreaData *areadata = areaattr;
   AreaData *childdata = childattr;

   areadata->Area += childdata->Area;
} /* MergeAreaData */

double AreaAttribute(void *areaattr)
{
   AreaData *areadata = areaattr;
   double area;

   area = areadata->Area;
   return(area);
} /* AreaAttribute */



/****** Typedefs and functions for minimum enclosing rectangle attributes *******/

typedef struct EnclRectData
{
   int MinX;
   int MinY;
   int MaxX;
   int MaxY;
} EnclRectData;

void *NewEnclRectData(int x, int y)
{
   EnclRectData *rectdata;

   rectdata = malloc(sizeof(EnclRectData));
   rectdata->MinX = rectdata->MaxX = x;
   rectdata->MinY = rectdata->MaxY = y;
   return(rectdata);
} /* NewEnclRectData */

void DeleteEnclRectData(void *rectattr)
{
   free(rectattr);
} /* DeleteEnclRectData */

void AddToEnclRectData(void *rectattr, int x, int y)
{
   EnclRectData *rectdata = rectattr;

   rectdata->MinX = MIN(rectdata->MinX, x);
   rectdata->MinY = MIN(rectdata->MinY, y);
   rectdata->MaxX = MAX(rectdata->MaxX, x);
   rectdata->MaxY = MAX(rectdata->MaxY, y);
} /* AddToEnclRectData */

void MergeEnclRectData(void *rectattr, void *childattr)
{
   EnclRectData *rectdata = rectattr;
   EnclRectData *childdata = childattr;

   rectdata->MinX = MIN(rectdata->MinX, childdata->MinX);
   rectdata->MinY = MIN(rectdata->MinY, childdata->MinY);
   rectdata->MaxX = MAX(rectdata->MaxX, childdata->MaxX);
   rectdata->MaxY = MAX(rectdata->MaxY, childdata->MaxY);
} /* MergeEnclRectData */

double EnclRectAreaAttribute(void *rectattr)
{
   EnclRectData *rectdata = rectattr;
   double area;

   area = (rectdata->MaxX - rectdata->MinX + 1)
        * (rectdata->MaxY - rectdata->MinY + 1);
   return(area);
} /* EnclRectAreaAttribute */

double EnclRectDiagAttribute(void *rectattr)
/* Computes the square of the length of the diagonal */
{
   EnclRectData *rectdata = rectattr;
   double l;

   l = (rectdata->MaxX - rectdata->MinX + 1)
     * (rectdata->MaxX - rectdata->MinX + 1)
     + (rectdata->MaxY - rectdata->MinY + 1)
     * (rectdata->MaxY - rectdata->MinY + 1);
   return(l);
} /* EnclRectDiagAttribute */



/****** Typedefs and functions for perimeter attributes *********************/

typedef struct PeriData
{
   long Area;
   long Perimeter;
} PeriData;

void *NewPeriData(int x, int y)
{
   PeriData *peridata;
   long neighbors[CONNECTIVITY];
   long lx=x, ly=y, p, peri=0, q;
   int numneighbors, i;
   ubyte h;

   p = ly*ImageWidth + lx;
   h = ORI[p];
   numneighbors = GetNeighbors(p, neighbors);
   peri += CONNECTIVITY-numneighbors;
   for (i=0; i<numneighbors; i++)
   {
      q = neighbors[i];
      if (ORI[q]<h)  peri++;
      if (ORI[q]>h)  peri--;
   }
   peridata = malloc(sizeof(PeriData));
   peridata->Area = 1;
   peridata->Perimeter = peri;
   return(peridata);
} /* NewPeriData */

void DeletePeriData(void *periattr)
{
   free(periattr);
} /* DeletePeriData */

void AddToPeriData(void *periattr, int x, int y)
{
   PeriData *peridata = periattr;
   long neighbors[CONNECTIVITY];
   long lx=x, ly=y, p, peri=0, q;
   int numneighbors, i;
   ubyte h;

   p = ly*ImageWidth + lx;
   h = ORI[p];
   numneighbors = GetNeighbors(p, neighbors);
   peri += CONNECTIVITY-numneighbors;
   for (i=0; i<numneighbors; i++)
   {
      q = neighbors[i];
      if (ORI[q]<h)  peri++;
      if (ORI[q]>h)  peri--;
   }
   peridata->Area ++;
   peridata->Perimeter += peri;
} /* AddToPeriData */

void MergePeriData(void *periattr, void *childattr)
{
   PeriData *peridata = periattr;
   PeriData *childdata = childattr;

   peridata->Area += childdata->Area;
   peridata->Perimeter += childdata->Perimeter;
} /* MergePeriData */

double PeriAreaAttribute(void *periattr)
{
   PeriData *peridata = periattr;
   double area;

   area = peridata->Area;
   return(area);
} /* PeriAreaAttribute */

double PeriPerimeterAttribute(void *periattr)
{
   PeriData *peridata = periattr;
   double peri;

   peri = peridata->Perimeter;
   return(peri);
} /* PeriPerimeterAttribute */

double PeriComplexityAttribute(void *periattr)
{
   PeriData *peridata = periattr;
   double area, peri;

   area = peridata->Area;
   peri = peridata->Perimeter;
   return(peri/area);
} /* PeriComplexityAttribute */

double PeriSimplicityAttribute(void *periattr)
{
   PeriData *peridata = periattr;
   double area, peri;

   area = peridata->Area;
   peri = peridata->Perimeter;
   return(area/peri);
} /* PeriSimplicityAttribute */

double PeriCompactnessAttribute(void *periattr)
{
   PeriData *peridata = periattr;
   double area, peri;

   area = peridata->Area;
   peri = peridata->Perimeter;
   return((peri*peri)/(4.0*PI*area));
} /* PeriCompactnessAttribute */



/****** Typedefs and functions for moment of inertia attributes **************************/

typedef struct InertiaData
{
   long Area;
   double SumX, SumY, SumX2, SumY2;
} InertiaData;

void *NewInertiaData(int x, int y)
{
   InertiaData *inertiadata;

   inertiadata = malloc(sizeof(InertiaData));
   inertiadata->Area = 1;
   inertiadata->SumX = x;
   inertiadata->SumY = y;
   inertiadata->SumX2 = x*x;
   inertiadata->SumY2 = y*y;
   return(inertiadata);
} /* NewInertiaData */

void DeleteInertiaData(void *inertiaattr)
{
   free(inertiaattr);
} /* DeleteInertiaData */

void AddToInertiaData(void *inertiaattr, int x, int y)
{
   InertiaData *inertiadata = inertiaattr;

   inertiadata->Area ++;
   inertiadata->SumX += x;
   inertiadata->SumY += y;
   inertiadata->SumX2 += x*x;
   inertiadata->SumY2 += y*y;
} /* AddToInertiaData */

void MergeInertiaData(void *inertiaattr, void *childattr)
{
   InertiaData *inertiadata = inertiaattr;
   InertiaData *childdata = childattr;

   inertiadata->Area += childdata->Area;
   inertiadata->SumX += childdata->SumX;
   inertiadata->SumY += childdata->SumY;
   inertiadata->SumX2 += childdata->SumX2;
   inertiadata->SumY2 += childdata->SumY2;
} /* MergeInertiaData */

double InertiaAttribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double inertia;

   inertia = inertiadata->SumX2 + inertiadata->SumY2 -
             (inertiadata->SumX * inertiadata->SumX +
              inertiadata->SumY * inertiadata->SumY) / (double)(inertiadata->Area)
             + (double)inertiadata->Area / 6.0;
   return(inertia);
} /* InertiaAttribute */

double InertiaDivA2Attribute(void *inertiaattr)
{
   InertiaData *inertiadata = inertiaattr;
   double inertia, area;

   area = (double)(inertiadata->Area);
   inertia = inertiadata->SumX2 + inertiadata->SumY2 -
             (inertiadata->SumX * inertiadata->SumX +
              inertiadata->SumY * inertiadata->SumY) / area
             + area / 6.0;
   return(inertia*2.0*PI/(area*area));
} /* InertiaDivA2Attribute */



/****** Typedefs and functions for jaggedness attributes *********************/

typedef struct JaggedData
{
   long Area;
   long Perimeter;
   double SumX, SumY, SumX2, SumY2;
} JaggedData;

void *NewJaggedData(int x, int y)
{
   JaggedData *jaggeddata;
   long neighbors[CONNECTIVITY];
   long lx=x, ly=y, p, peri=0, q;
   int numneighbors, i;
   ubyte h;

   p = ly*ImageWidth + lx;
   h = ORI[p];
   numneighbors = GetNeighbors(p, neighbors);
   peri += CONNECTIVITY-numneighbors;
   for (i=0; i<numneighbors; i++)
   {
      q = neighbors[i];
      if (ORI[q]<h)  peri++;
      if (ORI[q]>h)  peri--;
   }
   jaggeddata = malloc(sizeof(JaggedData));
   jaggeddata->Area = 1;
   jaggeddata->Perimeter = peri;
   jaggeddata->SumX = x;
   jaggeddata->SumY = y;
   jaggeddata->SumX2 = x*x;
   jaggeddata->SumY2 = y*y;
   return(jaggeddata);
} /* NewJaggedData */

void DeleteJaggedData(void *jaggedattr)
{
   free(jaggedattr);
} /* DeleteJaggedData */

void AddToJaggedData(void *jaggedattr, int x, int y)
{
   JaggedData *jaggeddata = jaggedattr;
   long neighbors[CONNECTIVITY];
   long lx=x, ly=y, p, peri=0, q;
   int numneighbors, i;
   ubyte h;

   p = ly*ImageWidth + lx;
   h = ORI[p];
   numneighbors = GetNeighbors(p, neighbors);
   peri += CONNECTIVITY-numneighbors;
   for (i=0; i<numneighbors; i++)
   {
      q = neighbors[i];
      if (ORI[q]<h)  peri++;
      if (ORI[q]>h)  peri--;
   }
   jaggeddata->Area ++;
   jaggeddata->Perimeter += peri;
   jaggeddata->SumX += x;
   jaggeddata->SumY += y;
   jaggeddata->SumX2 += x*x;
   jaggeddata->SumY2 += y*y;
} /* AddToJaggedData */

void MergeJaggedData(void *jaggedattr, void *childattr)
{
   JaggedData *jaggeddata = jaggedattr;
   JaggedData *childdata = childattr;

   jaggeddata->Area += childdata->Area;
   jaggeddata->Perimeter += childdata->Perimeter;
   jaggeddata->SumX += childdata->SumX;
   jaggeddata->SumY += childdata->SumY;
   jaggeddata->SumX2 += childdata->SumX2;
   jaggeddata->SumY2 += childdata->SumY2;
} /* MergeJaggedData */

double JaggedAttribute(void *jaggedattr)
{
   JaggedData *jaggeddata = jaggedattr;
   double peri;

   peri = jaggeddata->Perimeter;
   return(peri);
} /* JaggedPerimeterAttribute */

double JaggedCompactnessAttribute(void *jaggedattr)
{
   JaggedData *jaggeddata = jaggedattr;
   double area, peri;

   area = jaggeddata->Area;
   peri = jaggeddata->Perimeter;
   return((peri*peri)/(4.0*PI*area));
} /* JaggedCompactnessAttribute */

double JaggedInertiaDivA2Attribute(void *jaggedattr)
{
   JaggedData *jaggeddata = jaggedattr;
   double inertia, area;

   area = (double)(jaggeddata->Area);
   inertia = jaggeddata->SumX2 + jaggeddata->SumY2 -
             (jaggeddata->SumX * jaggeddata->SumX +
              jaggeddata->SumY * jaggeddata->SumY) / area
             + area / 6.0;
   return(inertia*2.0*PI/(area*area));
} /* JaggedInertiaDivA2Attribute */

double JaggednessAttribute(void *jaggedattr)
{
   JaggedData *jaggeddata = jaggedattr;
   double area, peri, inertia;

   area = (double)(jaggeddata->Area);
   peri = jaggeddata->Perimeter;
   inertia = jaggeddata->SumX2 + jaggeddata->SumY2 -
             (jaggeddata->SumX * jaggeddata->SumX +
              jaggeddata->SumY * jaggeddata->SumY) / area
             + area / 6.0;
   return(area*peri*peri/(8.0*PI*PI*inertia));
} /* JaggednessAttribute */



/****** Typedefs and functions for Entropy attributes ******************************/

typedef struct EntropyData
{
   long Hist[NUMLEVELS];
} EntropyData;

void *NewEntropyData(int x, int y)
{
   EntropyData *entropydata;
   long lx=x, ly=y, p;
   int i;

   p = ly*ImageWidth + lx;
   entropydata = malloc(sizeof(EntropyData));
   for (i=0; i<NUMLEVELS; i++)  entropydata->Hist[i] = 0;
   entropydata->Hist[ORI[p]] = 1;
   return(entropydata);
} /* NewEntropyData */

void DeleteEntropyData(void *entropyattr)
{
   free(entropyattr);
} /* DeleteEntropyData */

void AddToEntropyData(void *entropyattr, int x, int y)
{
   EntropyData *entropydata = entropyattr;
   long lx=x, ly=y, p;

   p = ly*ImageWidth + lx;
   entropydata->Hist[ORI[p]] ++;
} /* AddToEntropyData */

void MergeEntropyData(void *entropyattr, void *childattr)
{
   EntropyData *entropydata = entropyattr;
   EntropyData *childdata = childattr;
   int i;

   for (i=0; i<NUMLEVELS; i++)  entropydata->Hist[i] += childdata->Hist[i];
} /* MergeEntropyData */

double EntropyAttribute(void *entropyattr)
{
   EntropyData *entropydata = entropyattr;
   double p[NUMLEVELS];
   double num=0.0, entropy = 0.0;
   int i;

   for (i=0; i<NUMLEVELS; i++)  num += entropydata->Hist[i];
   for (i=0; i<NUMLEVELS; i++)  p[i] = (entropydata->Hist[i])/num;
   for (i=0; i<NUMLEVELS; i++)  entropy += p[i] * (log(p[i]+0.00001)/log(2.0));
   return(-entropy);
} /* EntropyAttribute */



/****** Typedefs and functions for lambda-max attributes *************************/

typedef struct LambdamaxData
{
   int MinLevel;
   int MaxLevel;
} LambdamaxData;

void *NewLambdamaxData(int x, int y)
{
   LambdamaxData *lambdadata;

   lambdadata = malloc(sizeof(LambdamaxData));
   lambdadata->MinLevel = ORI[y*ImageWidth+x];
   lambdadata->MaxLevel = ORI[y*ImageWidth+x];
   return(lambdadata);
} /* NewLambdamaxData */

void DeleteLambdamaxData(void *lambdaattr)
{
   free(lambdaattr);
} /* DeleteLambdamaxData */

void AddToLambdamaxData(void *lambdaattr, int x, int y)
{
} /* AddToLambdamaxData */

void MergeLambdamaxData(void *lambdaattr, void *childattr)
{
   LambdamaxData *lambdadata = lambdaattr;
   LambdamaxData *childdata = childattr;

   if (childdata->MaxLevel > lambdadata->MaxLevel)  lambdadata->MaxLevel = childdata->MaxLevel;
} /* MergeLambdamaxData */

double LambdamaxAttribute(void *lambdaattr)
{
   LambdamaxData *lambdadata = lambdaattr;
   double height;

   height = lambdadata->MaxLevel - lambdadata->MinLevel;
   return(height);
} /* LambdamaxAttribute */



/****** Typedefs and functions for pos attributes ******************************/

typedef struct PosData
{
   int xmax;
   int ymax;
} PosData;

void *NewPosData(int x, int y)
{
   PosData *posdata;

   posdata = malloc(sizeof(PosData));
   posdata->xmax = x;
   posdata->ymax = y;
   return(posdata);
} /* NewPosData */

void DeletePosData(void *posattr)
{
   free(posattr);
} /* DeletePosData */

void AddToPosData(void *posattr, int x, int y)
{
   PosData *posdata = posattr;
   unsigned long lx1, ly1, lx2, ly2, d1, d2;

   lx1 = posdata->xmax;
   ly1 = posdata->ymax;
   lx2 = x;
   ly2 = y;
   d1 = lx1*lx1+ly1*ly1;
   d2 = lx2*lx2+ly2*ly2;
   if (d2>d1)
   {
      posdata->xmax = x;
      posdata->ymax = y;
   }
} /* AddToPosData */

void MergePosData(void *posattr, void *childattr)
{
} /* MergePosData */

double PosXAttribute(void *posattr)
{
   PosData *posdata = posattr;
   double pos;

   pos = posdata->xmax;
   return(pos);
} /* PosXAttribute */

double PosYAttribute(void *posattr)
{
   PosData *posdata = posattr;
   double pos;

   pos = posdata->ymax;
   return(pos);
} /* PosYAttribute */



/****** Typedefs and functions for graylevel attributes ******************************/

typedef struct LevelData
{
   int level;
} LevelData;

void *NewLevelData(int x, int y)
{
   LevelData *leveldata;

   leveldata = malloc(sizeof(LevelData));
   leveldata->level = ORI[y*ImageWidth+x];
   return(leveldata);
} /* NewLevelData */

void DeleteLevelData(void *levelattr)
{
   free(levelattr);
} /* DeleteLevelData */

void AddToLevelData(void *levelattr, int x, int y)
{
} /* AddToLevelData */

void MergeLevelData(void *levelattr, void *childattr)
{
} /* MergeLevelData */

double LevelAttribute(void *levelattr)
{
   LevelData *leveldata = levelattr;
   double level;

   level = leveldata->level;
   return(level);
} /* LevelAttribute */

/****** Typedefs and functions for sum flux attributes ******************************/

typedef struct SumFluxData
{
  double SumFlux;
  long Area;
  int MinLevel;
} SumFluxData;

void *NewSumFluxData(int x, int y)
{
   SumFluxData *sumfluxdata;

   sumfluxdata = malloc(sizeof(SumFluxData));
   sumfluxdata->SumFlux = ORI[y*ImageWidth+x];
   sumfluxdata->MinLevel = ORI[y*ImageWidth+x];   
   sumfluxdata->Area = 1;
   return(sumfluxdata);
} /* NewSumFluxData */

void DeleteSumFluxData(void *sumfluxattr)
{
   free(sumfluxattr);
} /* DeleteSumFluxData */

void AddToSumFluxData(void *sumfluxattr, int x, int y)
{  
  SumFluxData *sumfluxdata = sumfluxattr; 
  sumfluxdata->SumFlux += ORI[y*ImageWidth+x];
  sumfluxdata->Area ++;
} /* AddToLevelData */

void MergeSumFluxData(void *sumfluxattr, void *childattr)
{
  SumFluxData *sumfluxdata = sumfluxattr, *childdata=childattr; 
  sumfluxdata->SumFlux += childdata->SumFlux;
  sumfluxdata->Area += childdata->Area;

} /* MergeLevelData */

double SumFluxAttribute(void *sumfluxattr)
{
  SumFluxData *sumfluxdata = sumfluxattr; 

  return
    (double)(sumfluxdata->SumFlux)/(double)(sumfluxdata->Area) - (double)(sumfluxdata->MinLevel);
} /* LevelAttribute */

