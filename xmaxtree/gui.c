/* gui.c */
/* October 2000  Erik Urbach */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "maxtree.h"
#include "mtmacros.h"
#include "ran3.h"
#include "gui.h"
#include "tlgui.h"



#define ButtonQuit      1
#define ButtonFilter    2
#define ButtonGran      3
#define ButtonInvert    4
#define ButtonNoise     5
#define ButtonAbout     6
#define IntDecision    11
#define IntAttr1       12
#define IntAttr2       13
#define IntWidth       14
#define IntHeight      15
#define IntMapper1     16
#define IntMapper2     17
#define FloatMin1      21
#define FloatMax1      22
#define FloatMin2      23
#define FloatMax2      24
#define FloatLambda1   25
#define FloatLambda2   26

#define ButtonNoiseAdd    101
#define ButtonNoiseMult   102
#define ButtonNoiseClose  103

#define ButtonAboutOk  201



typedef struct GUINoiseInfo GUINoiseInfo;

typedef struct GUIInfo
{
   MaxTree Tree;
   int Filter;
   int k;
   int mingreylevel;
   int NumAttrs;
   ProcSet *Procs;
   int *Dims;
   double *DLs;
   double *DHs;
   double *Result;
   TLDisplay *Display;
   TLWindow *ControlWindow;
   TLGadget *InputDecision;
   TLGadget *InputAttr1;
   TLGadget *InputAttr2;
   TLGadget *InputWidth;
   TLGadget *InputHeight;
   TLGadget *InputMapper1;
   TLGadget *InputMapper2;
   TLGadget *InputMin1;
   TLGadget *InputMax1;
   TLGadget *InputMin2;
   TLGadget *InputMax2;
   TLGadget *InputLambda1;
   TLGadget *InputLambda2;
   TLWindow *DiatomWindow;
   TLImage *DiatomImage;
   ubyte *DiatomPGM;
   ubyte *ShapePGM;
   long DiatomWidth;
   long DiatomHeight;
   TLWindow *GranWindow;
   TLImage *GranImage;
   ubyte *GranPGM;
   GUINoiseInfo *NoiseInfo;
   TLWindow *AboutWindow;
} GUIInfo;



struct GUINoiseInfo
{
   TLWindow *Window;
   TLGadget *InputMin;
   TLGadget *InputMax;
   TLGadget *InputPercent;
   TLGadget *InputWeight;
   TLGadget *ButtonOk;
};



void GUIFilterMDMin(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs,
                    int d1, int d2, int x, int y);
void GUIFilterMDSub(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs,
                    int d1, int d2, int x, int y, int mingreylevel);
void GUIFilterMDDirect(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, int *selcoord);
void GUIFilterMDMax(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, int *selcoord);
void TLImageSelectPixel(TLWindow *window, TLImage *image, int x, int y);
void Show2DSelGranNodes(GUIInfo *info, int d1, int d2, int x, int y);
double *GUIAllocResultArray(int numattrs, int *dims);
void GUIComputeGranulometry(GUIInfo *guiinfo);
void GUIInitGranulometryImage(GUIInfo *info);
void GUICreateGranulometryImage(GUIInfo *info);
GUIInfo *GUIInfoCreate(ubyte *diatompgm, ubyte *shapepgm, long width, long height, MaxTree tree, int filter, int k,
                       int mingreylevel, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, 
		       bool sharedcm);
void GUIInfoDelete(GUIInfo *info);
void GUIInvertDiatom(GUIInfo *info);
void GUIAddNoiseDiatom(GUIInfo *info, float min, float max, float perc, float weight);
void GUIMultNoiseDiatom(GUIInfo *info, float min, float max, float perc, float weight);
void GUINoiseDiatom(GUIInfo *info, int type, float min, float max, float perc, float weight);
int GUIOpenNoiseWindow(GUIInfo *info);
void GUICloseNoiseWindow(GUIInfo *info);
int GUIOpenAboutWindow(GUIInfo *info);
void GUICloseAboutWindow(GUIInfo *info);
void GUISetInputValues(GUIInfo *info);
int GUIOpenControlWindow(GUIInfo *info, char *title, char *iconname);
void GUICloseControlWindow(GUIInfo *info);
int GUIOpenDiatomWindow(GUIInfo *info, char *title, char *iconname);
void GUICloseDiatomWindow(GUIInfo *info);
int GUIOpenGranWindow(GUIInfo *info, char *title, char *iconname);
void GUICloseGranWindow(GUIInfo *info);




void GUIFilterMDMin(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs,
                    int d1, int d2, int x, int y)
{
   void **attrs;
   long i, idx, parent;
   int l, lm1, lm2;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            if (t[parent].NodeStatus)  t[idx].NodeStatus = 2;
            attrs = t[idx].Attribute;
            lm1 = procs[d1].Mapper(procs[d1].Attribute(attrs[d1]), dims[d1], dls[d1], dhs[d1]);
            lm2 = procs[d2].Mapper(procs[d2].Attribute(attrs[d2]), dims[d2], dls[d2], dhs[d2]);
            if ((lm1==x) && (lm2==y))  t[idx].NodeStatus = 1;
	 }
      }
   }
} /* GUIFilterMDMin */

void GUIFilterMDSub(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs,
                    int d1, int d2, int x, int y, int mingreylevel)
{
   void **attrs;
   long i, idx, parent;
   int l, lm1, lm2;

   for (l=mingreylevel; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            if (t[parent].NodeStatus & 3 )  t[idx].NodeStatus |= 2;
            attrs = t[idx].Attribute;
            lm1 = procs[d1].Mapper(procs[d1].Attribute(attrs[d1]), dims[d1], dls[d1], dhs[d1]);
            lm2 = procs[d2].Mapper(procs[d2].Attribute(attrs[d2]), dims[d2], dls[d2], dhs[d2]);
            if ((lm1==x) && (lm2==y) && t[parent].NodeStatus & 4 )  t[idx].NodeStatus |= 1;
	 }
      }
   }
} /* GUIFilterMDMin */



void GUIFilterMDDirect(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, int *selcoord)
{
   void **attrs;
   double d;
   long i, idx, parent, pos, area, maxpos, selpos;
   int l, j, lm;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         parent = t[idx].Parent;
         if (idx!=parent)
         {
            t[idx].NodeStatus = t[idx].Area;
            attrs = t[idx].Attribute;
	    pos = 0;
            selpos = 0;
            for (j=numattrs-1; j>=0; j--)
            {
               pos = pos*(dims[j]+1) + procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
               selpos = selpos*(dims[j]+1) + selcoord[j];
            }
            if (pos==selpos)  t[idx].NodeStatus = -t[idx].NodeStatus;
         }
      }
   }
   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         parent = t[idx].Parent;
         area = abs(t[idx].NodeStatus);
         attrs = t[idx].Attribute;
         for (j=numattrs-1; j>=0; j--)
         {
            t[parent].Pos[j] = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
         }
         idx = parent;
         parent = t[parent].Parent;
         while (idx!=parent)
         {
            attrs = t[idx].Attribute;
            pos = 0;
            maxpos = 0;
            selpos = 0;
            for (j=numattrs-1; j>=0; j--)
            {
               lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
               t[parent].Pos[j] = MAX(lm, t[idx].Pos[j]);
               pos = pos*(dims[j]+1) + lm;
               maxpos = maxpos*(dims[j]+1) + t[parent].Pos[j];
               selpos = selpos*(dims[j]+1) + selcoord[j];
            }
	    if (maxpos>pos)
	    {
               d = area*(t[idx].Level-t[parent].Level);
               if ((selpos==maxpos) && (t[parent].NodeStatus>0))  t[parent].NodeStatus = -t[parent].NodeStatus;
               if (t[idx].NodeStatus>0)  t[idx].NodeStatus -= area;
	       else  t[idx].NodeStatus += area;
               idx = parent;
               parent = t[parent].Parent;
            } else  idx=parent;
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         if (t[idx].NodeStatus < 0)  t[idx].NodeStatus = 1;
         else t[idx].NodeStatus = 0;
      }
   }
} /* GUIFilterMDDirect */



void GUIFilterMDMax(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, int *selcoord)
{
   void **attrs;
   long i, idx, parent, pos, selpos;
   int l, j, lm;

   for (l=NUMLEVELS-1; l>=0; l--)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         parent = t[idx].Parent;
	 if (idx!=parent)
         {
            attrs = t[idx].Attribute;
            pos = 0;
            selpos = 0;
            for (j=numattrs-1; j>=0; j--)
	    {
               lm = procs[j].Mapper(procs[j].Attribute(attrs[j]), dims[j], dls[j], dhs[j]);
	       if (t[idx].NodeStatus)  lm = MAX(lm, t[idx].Pos[j]);
	       if (t[parent].NodeStatus)  t[parent].Pos[j] = MAX(t[parent].Pos[j], lm);
               else  t[parent].Pos[j] = lm;
               pos *= dims[j] + 1;
               pos += lm;
               selpos = selpos*(dims[j]+1) + selcoord[j];
	    }
            if (pos==selpos)  t[idx].NodeStatus = 1;
            else  t[idx].NodeStatus = 0;
            t[parent].NodeStatus = 1;
	 }
      }
   }
} /* GUIFilterMDMax */



void TLImageSelectPixel(TLWindow *window, TLImage *image, int x, int y)
{
   int a, b;

   for (b=0; b<8; b++)
   {
      for (a=0; a<8; a++)
      {
         XPutPixel(image->Image, x*8+a, y*8+b, window->Display->Red.pixel);
      }
   }
} /* TLImageSelectPixel */



void Show2DSelGranNodes(GUIInfo *info, int d1, int d2, int x, int y)
{
   int *selcoord;
   long i, idx, lx, ly, p;
   int l;

   selcoord = calloc(info->NumAttrs, sizeof(int));
   for (i=0; i<info->NumAttrs; i++)  selcoord[i] = 0;
   selcoord[d1] = x;
   selcoord[d2] = y;
   if (info->Filter==0)  GUIFilterMDMin(info->Tree, info->NumAttrs, info->Procs, info->Dims, info->DLs, info->DHs, d1, d2, x, y);
   else if (info->Filter==1)  GUIFilterMDDirect(info->Tree, info->NumAttrs, info->Procs, info->Dims, info->DLs, info->DHs, selcoord);
   else if (info->Filter==2)  GUIFilterMDMax(info->Tree, info->NumAttrs, info->Procs, info->Dims, info->DLs, info->DHs, selcoord);
   else if (info->Filter==3)  GUIFilterMDSub(info->Tree, info->NumAttrs, info->Procs, info->Dims,
                                             info->DLs, info->DHs, d1, d2, x, y, info->mingreylevel);
   free(selcoord);
   for (ly=0; ly<info->DiatomHeight; ly++)
   {
      for (lx=0; lx<info->DiatomWidth; lx++)
      {
         p = ly*(info->DiatomWidth)+lx;
	 if (info->ShapePGM[p])
	 {
            idx = GetNodeIndex(p, info->DiatomPGM[p]);
            if (info->Tree[idx].NodeStatus & 1)  XPutPixel(info->DiatomImage->Image, lx, ly, info->Display->Red.pixel);
            else if (info->Tree[idx].NodeStatus & 2)  XPutPixel(info->DiatomImage->Image, lx, ly, info->Display->Orange.pixel);
         }
      }
   }
   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
         info->Tree[idx].NodeStatus &= 4;
      }
   }
} /*  Show2DSelGranNodes */



double *GUIAllocResultArray(int numattrs, int *dims)
{
   long resultsize;
   int x;

   resultsize = dims[0]+1;
   for (x=1; x<numattrs; x++)  resultsize *= dims[x]+1;
   return(calloc((size_t)resultsize, sizeof(double)));
} /* GUIAllocResultArray */



void GUIComputeGranulometry(GUIInfo *guiinfo)
{
   MaxTree tree;
   int filter, k, numattrs, *dims;
   ProcSet *procs;
   double *result, *dls, *dhs;

   tree = guiinfo->Tree;
   filter = guiinfo->Filter;
   numattrs = guiinfo->NumAttrs;
   procs = guiinfo->Procs;
   dims = guiinfo->Dims;
   dls = guiinfo->DLs;
   dhs = guiinfo->DHs;
   k = guiinfo->k;
   result = guiinfo->Result;
   if (filter==0)       MaxTreeGranulometryMDMin(tree, numattrs, procs, dims, result, dls, dhs);
   else if (filter==1)  MaxTreeGranulometryMDDirect(tree, numattrs, procs, dims, result, dls, dhs);
   else if (filter==2)  MaxTreeGranulometryMDMax(tree, numattrs, procs, dims, result, dls, dhs);
   else if (filter==3)  MaxTreeGranulometryMDWilkinsonK(tree, numattrs, procs, dims, result, dls, dhs, k);
} /* GUIComputeGranulometry */



void GUIInitGranulometryImage(GUIInfo *info)
{
   double gmax;
   int x, y, a, b;

   gmax = log(1.0+info->Result[0]);
   for (y=0; y<info->Dims[1]; y++)
   {
      for (x=0; x<info->Dims[0]; x++)
      {
         gmax = MAX(gmax, log(1.0+info->Result[y*(info->Dims[0]+1)+x]));
      }
   }
   for (y=0; y<info->Dims[1]; y++)
   {
      for (x=0; x<info->Dims[0]; x++)
      {
         for (b=0; b<8; b++)
         {
            for (a=0; a<8; a++)
            {
               info->GranPGM[(y*8+b)*8*(info->Dims[0])+x*8+a] = (log(1.0+info->Result[y*(info->Dims[0]+1)+x])/gmax)*255.0;
            }
         }
      }
   }
} /* GUIInitGranulometryImage */



void GUICreateGranulometryImage(GUIInfo *info)
{
   info->GranPGM = malloc((info->Dims[0])*8*(info->Dims[1])*8);
   if (info->GranPGM)  GUIInitGranulometryImage(info);
} /* GUICreateGranulometryImage */



GUIInfo *GUIInfoCreate(ubyte *diatompgm, ubyte *shapepgm, long width, long height, MaxTree tree, int filter, int k,
                       int mingreylevel, int numattrs, ProcSet *procs, int *dims, double *dls, double *dhs, bool sharedcm)
{
   GUIInfo *info;

   info = malloc(sizeof(GUIInfo));
   if (info==NULL)  return(NULL);
   info->Tree = tree;
   info->Filter = filter;
   info->k = k;
   info->mingreylevel = mingreylevel;
   info->NumAttrs = numattrs;
   info->Procs = procs;
   info->Dims = dims;
   info->DLs = dls;
   info->DHs = dhs;
   info->Result = GUIAllocResultArray(numattrs, dims);
   if (info->Result==NULL)
   {
      free(info);
      return(NULL);
   }
   info->DiatomPGM = diatompgm;
   info->DiatomWidth = width;
   info->DiatomHeight = height;
   info->ShapePGM = shapepgm;
   GUIComputeGranulometry(info);
   GUICreateGranulometryImage(info);
   if (info->GranPGM==NULL)
   {
      free(info->Result);
      free(info);
      return(NULL);
   }
   info->Display = TLCreateDisplay(sharedcm);
   if (info->Display==NULL)
   {
      free(info->GranPGM);
      free(info->Result);
      free(info);
      return(NULL);
   }
   info->ControlWindow = NULL;
   info->DiatomWindow = NULL;
   info->DiatomImage = NULL;
   info->GranWindow = NULL;
   info->GranImage = NULL;
   info->NoiseInfo = NULL;
   info->AboutWindow = NULL;
   return(info);
} /* GUIInfoCreate */



void GUIInfoDelete(GUIInfo *info)
{
   if (info->AboutWindow)  GUICloseAboutWindow(info);
   if (info->NoiseInfo)  GUICloseNoiseWindow(info);
   if (info->Result)  free(info->Result);
   if (info->GranImage)  TLDeleteImage(info->GranImage);
   if (info->GranWindow)  TLDeleteWindow(info->GranWindow);
   if (info->DiatomImage)  TLDeleteImage(info->DiatomImage);
   if (info->DiatomWindow)  TLDeleteWindow(info->DiatomWindow);
   if (info->ControlWindow)  TLDeleteWindow(info->ControlWindow);
   TLDeleteDisplay(info->Display);
   free(info->GranPGM);
   free(info);
} /* GUIInfoDelete */



void GUIInvertDiatom(GUIInfo *info)
{
   ulong p;

   for (p=0; p<(info->DiatomWidth * info->DiatomHeight); p++)   info->DiatomPGM[p] = 255-info->DiatomPGM[p];
   MaxTreeDestroyMD(info->NumAttrs, info->Procs);
   info->Tree = MaxTreeBuildMD(info->NumAttrs, info->Procs);
   GUIComputeGranulometry(info);
   GUIInitGranulometryImage(info);
   TLRefreshImage(info->DiatomWindow, info->DiatomImage, info->DiatomPGM);
   TLPutImage(info->DiatomWindow, info->DiatomImage);
   TLRefreshImage(info->GranWindow, info->GranImage, info->GranPGM);
   TLPutImage(info->GranWindow, info->GranImage);
} /* GUIInvertDiatom */



void GUIAddNoiseDiatom(GUIInfo *info, float min, float max, float perc, float weight)
{
   float f;
   ulong p;
   int iseed=0, v;

   for (p=0; p<(info->DiatomWidth * info->DiatomHeight); p++)
   {
      f = ran3(&iseed);
      if (f<perc)
      {
         f = ran3(&iseed);
         v = info->DiatomPGM[p] + min*255.0 + f*(max-min)*255.0 + 0.5;
         v = info->DiatomPGM[p]*(1.0-weight) + v*weight + 0.5;
	 if (v<0)  v=0;
	 if (v>255)  v=255;
         info->DiatomPGM[p] = v;
      }
   }
} /* GUIAddNoiseDiatom */



void GUIMultNoiseDiatom(GUIInfo *info, float min, float max, float perc, float weight)
{
   float f;
   ulong p;
   int iseed=0, v;

   for (p=0; p<(info->DiatomWidth * info->DiatomHeight); p++)
   {
      f = ran3(&iseed);
      if (f<perc)
      {
         f = ran3(&iseed);
         f = min+f*(max-min);
         v = info->DiatomPGM[p] * f + 0.5;
         v = info->DiatomPGM[p]*(1.0-weight) + v*weight + 0.5;
         if (v<0)  v=0;
	 if (v>255)  v=255;
         info->DiatomPGM[p] = v;
      }
   }
} /* GUIMultNoiseDiatom */



void GUINoiseDiatom(GUIInfo *info, int type, float min, float max, float perc, float weight)
{
   if (type==1)  GUIAddNoiseDiatom(info, min, max, perc, weight);
   if (type==2)  GUIMultNoiseDiatom(info, min, max, perc, weight);
   MaxTreeDestroyMD(info->NumAttrs, info->Procs);
   info->Tree = MaxTreeBuildMD(info->NumAttrs, info->Procs);
   GUIComputeGranulometry(info);
   GUIInitGranulometryImage(info);
   TLRefreshImage(info->DiatomWindow, info->DiatomImage, info->DiatomPGM);
   TLPutImage(info->DiatomWindow, info->DiatomImage);
   TLRefreshImage(info->GranWindow, info->GranImage, info->GranPGM);
   TLPutImage(info->GranWindow, info->GranImage);
} /* GUINoiseDiatom */



int GUIOpenNoiseWindow(GUIInfo *info)
{
   GUINoiseInfo *noise;
   TLWindow *window;

   if (info->NoiseInfo)  return(0);
   noise = info->NoiseInfo = malloc(sizeof(GUINoiseInfo));
   if (info->NoiseInfo==NULL)  return(-1);
   window = info->NoiseInfo->Window = TLCreateWindow(info->Display, 400, 200, "Noise", "MT:Noise");
   if (window==NULL)
   {
      free(info->NoiseInfo);
      return(-1);
   }
   TLCreateLabel(window, 30,30, "Min:");
   noise->InputMin = TLCreateFloatInput(window, 100,30,80,25, 10, 3, NULL);
   TLCreateLabel(window, 210,30, "Max:");
   noise->InputMax = TLCreateFloatInput(window, 280,30,80,25, 10, 3, NULL);

   TLCreateLabel(window, 30,70, "% Noise:");
   noise->InputPercent = TLCreateFloatInput(window, 100,70,80,25, 10, 3, NULL);
   TLCreateLabel(window, 210,70, "Weight:");
   noise->InputWeight = TLCreateFloatInput(window, 280,70,80,25, 10, 3, NULL);

   TLFloatInputSetValue(noise->InputMin, 0.0);
   TLFloatInputSetValue(noise->InputMax, 1.0);
   TLFloatInputSetValue(noise->InputPercent, 0.5);
   TLFloatInputSetValue(noise->InputWeight, 1.0);

   TLCreateButton(window, 50,140,60,25, "Add", ButtonNoiseAdd, NULL);
   TLCreateButton(window, 130,140,60,25, "Mult", ButtonNoiseMult, NULL);
   TLCreateButton(window, 210,140,60,25, "Close", ButtonNoiseClose, NULL);

   return(0);
} /* GUIOpenNoiseWindow */



void GUICloseNoiseWindow(GUIInfo *info)
{
   if (info->NoiseInfo)  TLDeleteWindow(info->NoiseInfo->Window);
   free(info->NoiseInfo);
   info->NoiseInfo = NULL;
} /* GUICloseNoiseWindow */



int GUIOpenAboutWindow(GUIInfo *info)
{
   TLWindow *window;
   char str[80];

   if (info->AboutWindow)  return(0);
   window = info->AboutWindow = TLCreateWindow(info->Display, 400, 200, "About", "MT:About");
   if (window==NULL)  return(-1);
   TLCreateLabel(window, 30,10, "xmaxtree");
   TLCreateLabel(window, 30,30, "October 2000 Erik Urbach");
   sprintf(str, "Display Depth = %d", TLDisplayGetDepth(info->Display));
   TLCreateLabel(window, 30,70, str);
   sprintf(str, "Display Cells = %d", TLDisplayGetDisplayCells(info->Display));
   TLCreateLabel(window, 30,90, str);
   sprintf(str, "Display Planes = %d", TLDisplayGetDisplayPlanes(info->Display));
   TLCreateLabel(window, 30,110, str);
   TLCreateButton(window, 130,150,60,25, "Ok", ButtonAboutOk, NULL);
   return(0);
} /* GUIOpenAboutWindow */



void GUICloseAboutWindow(GUIInfo *info)
{
   if (info->AboutWindow)  TLDeleteWindow(info->AboutWindow);
   info->AboutWindow = NULL;
} /* GUICloseAboutWindow */



void GUISetInputValues(GUIInfo *info)
{
   TLIntInputSetValue(info->InputDecision, info->Filter);
   TLIntInputSetValue(info->InputAttr1, -1);
   TLIntInputSetValue(info->InputAttr2, -1);
   TLIntInputSetValue(info->InputWidth, info->Dims[0]);
   TLIntInputSetValue(info->InputHeight, info->Dims[1]);
   TLIntInputSetValue(info->InputMapper1, -1);
   TLIntInputSetValue(info->InputMapper2, -1);
   TLFloatInputSetValue(info->InputMin1, info->DLs[0]);
   TLFloatInputSetValue(info->InputMax1, info->DHs[0]);
   TLFloatInputSetValue(info->InputMin2, info->DLs[1]);
   TLFloatInputSetValue(info->InputMax2, info->DHs[1]);
   TLFloatInputSetValue(info->InputLambda1, -1.0);
   TLFloatInputSetValue(info->InputLambda2, -1.0);
} /* GUISetInputValues */



int GUIOpenControlWindow(GUIInfo *info, char *title, char *iconname)
{
   TLWindow *window;

   if (info->ControlWindow)  return(0);
   window = info->ControlWindow = TLCreateWindow(info->Display, 500, 270, title, iconname);
   if (window==NULL)  return(-1);
   TLCreateLabel(window, 10,10, "Decision:");
   info->InputDecision = TLCreateIntInput(window, 90,10,40,25, 1, IntDecision, NULL);
   TLCreateLabel(window, 150,10, "Attr1:");
   info->InputAttr1 = TLCreateIntInput(window, 200,10,40,25, 2, IntAttr1, NULL);
   TLCreateLabel(window, 250,10, "Attr2:");
   info->InputAttr2 = TLCreateIntInput(window, 300,10,40,25, 2, IntAttr2, NULL);

   TLCreateButton(window, 10,50,110,25, "Granulometry", ButtonGran, NULL);
   TLCreateLabel(window, 140,50, "Width:");
   info->InputWidth = TLCreateIntInput(window, 220,50,80,25, 10, IntWidth, NULL);
   TLCreateLabel(window, 310,50, "Height:");
   info->InputHeight = TLCreateIntInput(window, 380,50,80,25, 10, IntHeight, NULL);

   TLCreateLabel(window, 140,80, "Mapper1:");
   info->InputMapper1 = TLCreateIntInput(window, 220,80,80,25, 10, IntMapper1, NULL);
   TLCreateLabel(window, 310,80, "Mapper2:");
   info->InputMapper2 = TLCreateIntInput(window, 380,80,80,25, 10, IntMapper2, NULL);

   TLCreateLabel(window, 140,110, "Min1:");
   info->InputMin1 = TLCreateFloatInput(window, 220,110,80,25, 10, FloatMin1, NULL);
   TLCreateLabel(window, 310,110, "Max1:");
   info->InputMax1 = TLCreateFloatInput(window, 380,110,80,25, 10, FloatMax1, NULL);

   TLCreateLabel(window, 140,140, "Min2:");
   info->InputMin2 = TLCreateFloatInput(window, 220,140,80,25, 10, FloatMin2, NULL);
   TLCreateLabel(window, 310,140, "Max2:");
   info->InputMax2 = TLCreateFloatInput(window, 380,140,80,25, 10, FloatMax2, NULL);

   TLCreateButton(window, 10,170,110,25, "Filter", ButtonGran, NULL);
   TLCreateLabel(window, 140,170, "Lambda1:");
   info->InputLambda1 = TLCreateFloatInput(window, 220,170,80,25, 10, FloatLambda1, NULL);
   TLCreateLabel(window, 310,170, "Lambda2:");
   info->InputLambda2 = TLCreateFloatInput(window, 380,170,80,25, 10, FloatLambda2, NULL);

   TLCreateButton(window, 70,220,60,25, "Invert", ButtonInvert, NULL);
   TLCreateButton(window, 160,220,60,25, "Noise", ButtonNoise, NULL);
   TLCreateButton(window, 250,220,60,25, "About", ButtonAbout, NULL);
   TLCreateButton(window, 340,220,60,25, "Quit", ButtonQuit, NULL);

   GUISetInputValues(info);
   return(0);
} /* GUIOpenControlWindow */



void GUICloseControlWindow(GUIInfo *info)
{
   if (info->ControlWindow)  TLDeleteWindow(info->ControlWindow);
   info->ControlWindow = NULL;
} /* GUICloseControlWindow */



int GUIOpenDiatomWindow(GUIInfo *info, char *title, char *iconname)
{
   if (info->DiatomWindow)  return(0);
   info->DiatomWindow = TLCreateWindow(info->Display, info->DiatomWidth, info->DiatomHeight, title, iconname);
   if (info->DiatomWindow==NULL)  return(-1);
   info->DiatomImage = TLCreateImage(info->DiatomWindow, info->DiatomPGM, info->DiatomWidth, info->DiatomHeight);
   if (info->DiatomImage==NULL)
   {
      TLDeleteWindow(info->DiatomWindow);
      info->DiatomWindow = NULL;
      return(-1);
   }
   TLWindowSelectInput(info->DiatomWindow, TLWAF_MouseMove | TLWAF_Refresh | TLWAF_Leave);
   return(0);
} /* GUIOpenDiatomWindow */



void GUICloseDiatomWindow(GUIInfo *info)
{
   if (info->DiatomImage)  TLDeleteImage(info->DiatomImage);
   if (info->DiatomWindow)  TLDeleteWindow(info->DiatomWindow);
   info->DiatomImage = NULL;
   info->DiatomWindow = NULL;
} /* GUICloseDiatomWindow */



int GUIOpenGranWindow(GUIInfo *info, char *title, char *iconname)
{
   if (info->GranWindow)  return(0);
   info->GranWindow = TLCreateWindow(info->Display, info->Dims[0]*8, info->Dims[1]*8, title, iconname);
   if (info->GranWindow==NULL)  return(-1);
   info->GranImage = TLCreateImage(info->GranWindow, info->GranPGM, info->Dims[0]*8, info->Dims[1]*8);
   if (info->GranImage==NULL)
   {
      TLDeleteWindow(info->GranWindow);
      info->GranWindow = NULL;
      return(-1);
   }
   TLWindowSelectInput(info->GranWindow, TLWAF_MousePress | TLWAF_MouseRelease | TLWAF_MouseMove | TLWAF_Refresh | TLWAF_Leave);
   return(0);
} /* GUIOpenGranWindow */



void GUICloseGranWindow(GUIInfo *info)
{
   if (info->GranImage)  TLDeleteImage(info->GranImage);
   if (info->GranWindow)  TLDeleteWindow(info->GranWindow);
   info->GranImage = NULL;
   info->GranWindow = NULL;
} /* GUICloseGranWindow */



bool HandleNoiseButtons(GUIInfo *guiinfo, int id, void *userdata)
{
   float min, max, perc, weight;

   min = TLFloatInputGetValue(guiinfo->NoiseInfo->InputMin);
   if (min<-1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMin, -1.0);
   if ((id==ButtonNoiseMult) && (min<0.0))  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMin, 0.0);
   if (min>1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMin, 1.0);
   max = TLFloatInputGetValue(guiinfo->NoiseInfo->InputMax);
   if (max<-1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMax, -1.0);
   if (max>1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMax, 1.0);
   if (max<min)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputMax, min);
   perc = TLFloatInputGetValue(guiinfo->NoiseInfo->InputPercent);
   if (perc<0.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputPercent, 0.0);
   if (perc>1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputPercent, 1.0);
   weight = TLFloatInputGetValue(guiinfo->NoiseInfo->InputWeight);
   if (weight<0.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputWeight, 0.0);
   if (weight>1.0)  TLFloatInputSetValue(guiinfo->NoiseInfo->InputWeight, 1.0);
   switch(id)
   {
      case ButtonNoiseAdd:
         GUINoiseDiatom(guiinfo, 1, min, max, perc, weight);
         break;
      case ButtonNoiseMult:
         GUINoiseDiatom(guiinfo, 2, min, max, perc, weight);
         break;
      case ButtonNoiseClose:
         GUICloseNoiseWindow(guiinfo);
         break;
   }
   return(false);
} /* HandleNoiseButtons */



bool HandleNoiseEvents(GUIInfo *guiinfo, TLEvent *event)
{
   bool result = false;

   switch(event->Type)
   {
      case TLET_Button:
         result = HandleNoiseButtons(guiinfo, event->ButtonEvent.ID, event->ButtonEvent.UserData);
         break;
   }
   return(result);
} /* HandleNoiseEvents */



bool HandleAboutEvents(GUIInfo *guiinfo, TLEvent *event)
{
   GUICloseAboutWindow(guiinfo);
   return(false);
} /* HandleAboutEvents */



bool HandleControlButtons(GUIInfo *guiinfo, int id, void *userdata)
{
   int v;
   bool result = false;
 
   v = TLIntInputGetValue(guiinfo->InputDecision);
   if ((v<0) || (v>3))  TLIntInputSetValue(guiinfo->InputDecision, guiinfo->Filter);
   else  guiinfo->Filter = v;
   v = TLIntInputGetValue(guiinfo->InputWidth);
   if (v<1)  TLIntInputSetValue(guiinfo->InputWidth, guiinfo->Dims[0]);
   else  guiinfo->Dims[0] = v;
   v = TLIntInputGetValue(guiinfo->InputHeight);
   if (v<1)  TLIntInputSetValue(guiinfo->InputHeight, guiinfo->Dims[1]);
   else  guiinfo->Dims[1] = v;
   guiinfo->DLs[0] = TLFloatInputGetValue(guiinfo->InputMin1);
   guiinfo->DHs[0] = TLFloatInputGetValue(guiinfo->InputMax1);
   guiinfo->DLs[1] = TLFloatInputGetValue(guiinfo->InputMin2);
   guiinfo->DHs[1] = TLFloatInputGetValue(guiinfo->InputMax2);
   switch(id)
   {
      case ButtonFilter:
         break;
      case ButtonGran:
         GUICloseGranWindow(guiinfo);
         free(guiinfo->GranPGM);
         free(guiinfo->Result);
         guiinfo->Result = GUIAllocResultArray(guiinfo->NumAttrs, guiinfo->Dims);
         GUIComputeGranulometry(guiinfo);
         GUICreateGranulometryImage(guiinfo);
         GUIOpenGranWindow(guiinfo, "Granulometry", "MT:Gran");
         TLRefreshImage(guiinfo->DiatomWindow, guiinfo->DiatomImage, guiinfo->DiatomPGM);
         TLPutImage(guiinfo->DiatomWindow, guiinfo->DiatomImage);
         TLRefreshImage(guiinfo->GranWindow, guiinfo->GranImage, guiinfo->GranPGM);
         TLPutImage(guiinfo->GranWindow, guiinfo->GranImage);
         break;
      case ButtonInvert:
         GUIInvertDiatom(guiinfo);
         break;
      case ButtonNoise:
         GUIOpenNoiseWindow(guiinfo);
         break;
      case ButtonAbout:
         GUIOpenAboutWindow(guiinfo);
         break;
      case ButtonQuit:
         result = true;
         break;
   }
   return(result);
} /* HandleControlButtons */



bool HandleControlIntInputs(GUIInfo *guiinfo, int id, void *userdata)
{
   int v;

   switch(id)
   {
      case IntDecision:
         v = TLIntInputGetValue(guiinfo->InputDecision);
	 if ((v<0) || (v>3))  TLIntInputSetValue(guiinfo->InputDecision, guiinfo->Filter);
         break;
      case IntWidth:
         v = TLIntInputGetValue(guiinfo->InputWidth);
	 if (v<1)  TLIntInputSetValue(guiinfo->InputWidth, guiinfo->Dims[0]);
         break;
      case IntHeight:
         v = TLIntInputGetValue(guiinfo->InputHeight);
	 if (v<1)  TLIntInputSetValue(guiinfo->InputHeight, guiinfo->Dims[1]);
         break;
   }
   return(false);
} /* HandleControlIntInputs */



bool HandleControlEvents(GUIInfo *guiinfo, TLEvent *event)
{
   bool result = false;

   switch(event->Type)
   {
      case TLET_Button:
         result = HandleControlButtons(guiinfo, event->ButtonEvent.ID, event->ButtonEvent.UserData);
         break;
      case TLET_IntInput:
         result = HandleControlIntInputs(guiinfo, event->ButtonEvent.ID, event->ButtonEvent.UserData);
         break;
   }
   return(result);
} /* HandleControlEvents */



void HandleDiatomEvents(GUIInfo *guiinfo, TLEvent *event)
{
   int x, y;
   char str[100];

   switch(event->Type)
   {
      case TLET_WindowRefresh:
         TLPutImage(guiinfo->DiatomWindow, guiinfo->DiatomImage);
         break;
      case TLET_WindowLeave:
         XStoreName(guiinfo->Display->Display, guiinfo->DiatomWindow->Window, "Diatom window");
         break;
      case TLET_Mouse: 
         x = event->MouseEvent.X;
         y = event->MouseEvent.Y;
         if ((x>=0) && (y>=0) && (x<guiinfo->DiatomWidth) && (y<guiinfo->DiatomHeight))  sprintf(str, "Diatom (%d, %d, %d)", x,y,guiinfo->DiatomPGM[y*(guiinfo->DiatomWidth)+x]);
         else  strcpy(str, "Diatom window");
         XStoreName(guiinfo->Display->Display, guiinfo->DiatomWindow->Window, str);
         break;
   }
} /* HandleDiatomEvents */



int prev_x=-1, prev_y=-1;
bool buttonpressed = false, outsidegran;

void HandleGranEvents(GUIInfo *guiinfo, TLEvent *event)
{
   int x, y;
   char str[100];

   switch(event->Type)
   {
      case TLET_WindowRefresh:
         TLPutImage(guiinfo->GranWindow, guiinfo->GranImage);
         break;
      case TLET_Mouse:
         switch(event->MouseEvent.Action)
         {
            case TLMEA_MousePress:
               x = (event->MouseEvent.X)/8;
               y = (event->MouseEvent.Y)/8;
               if ((x>=0) && (y>=0) && (x<guiinfo->Dims[0]) && (y<guiinfo->Dims[1]))
               {
                  buttonpressed = true;
                  prev_x = x; prev_y = y;
                  outsidegran = false;
                  TLRefreshImage(guiinfo->DiatomWindow, guiinfo->DiatomImage, guiinfo->DiatomPGM);
                  TLRefreshImage(guiinfo->GranWindow, guiinfo->GranImage, guiinfo->GranPGM);
                  Show2DSelGranNodes(guiinfo, 0, 1, x, y);
                  TLImageSelectPixel(guiinfo->GranWindow, guiinfo->GranImage, x, y);
                  TLPutImage(guiinfo->GranWindow, guiinfo->GranImage);
               }
               break;
            case TLMEA_MouseRelease:
               TLPutImage(guiinfo->DiatomWindow, guiinfo->DiatomImage);
               buttonpressed = false;
               prev_x = prev_y = -1;
               break;
            case TLMEA_MouseMove:
               x = (event->MouseEvent.X)/8;
               y = (event->MouseEvent.Y)/8;
               if ((x>=0) && (y>=0) && (x<guiinfo->Dims[0]) && (y<guiinfo->Dims[1]))
               {
                  outsidegran = false;
	          if ((prev_x!=x) || (prev_y!=y))
                  {
                     prev_x = x; prev_y = y;
                     sprintf(str, "Granulometry (%d, %d, %1.0f)", x,y,guiinfo->Result[y*(guiinfo->Dims[0]+1)+x]);
                     XStoreName(guiinfo->Display->Display, guiinfo->GranWindow->Window, str);
                     if (buttonpressed)
                     {
                        Show2DSelGranNodes(guiinfo, 0, 1, x, y);
                        TLImageSelectPixel(guiinfo->GranWindow, guiinfo->GranImage, x, y);
                        TLPutImage(guiinfo->GranWindow, guiinfo->GranImage);
                     }
	          }
               } else {
                  if (!outsidegran)
                  {
                     outsidegran = true;
                     XStoreName(guiinfo->Display->Display, guiinfo->GranWindow->Window, "Granulometry window");
                  }
	       }
               break;
         }
         break;
      case TLET_WindowLeave:
         outsidegran = true;
         XStoreName(guiinfo->Display->Display, guiinfo->GranWindow->Window, "Granulometry window");
         break;
   }
} /* HandleGranEvents */



void HandleEvents(GUIInfo *info)
{
   TLEvent event;
   bool done = false;

   while (!done)
   {
      TLEventWait(info->Display, &event);
      if ((info->DiatomWindow) && (event.AnyEvent.Window==info->DiatomWindow))  HandleDiatomEvents(info, &event);
      else if ((info->GranWindow) && (event.AnyEvent.Window==info->GranWindow))  HandleGranEvents(info, &event);
      else if ((info->ControlWindow) && (event.AnyEvent.Window==info->ControlWindow))  done = HandleControlEvents(info, &event);
      else if ((info->NoiseInfo) && (event.AnyEvent.Window==info->NoiseInfo->Window))  done = HandleNoiseEvents(info, &event);
      else if ((info->AboutWindow) && (event.AnyEvent.Window==info->AboutWindow))  done = HandleAboutEvents(info, &event);
   }
} /* HandleEvents */



int GUIShow(ubyte *img, ubyte *shape, long width, long height, MaxTree tree, int filter, int k, int mingreylevel, 
	    int numattrs,
            ProcSet *procs, int *dims, double *dls, double *dhs, bool sharedcm)
{
   GUIInfo *guiinfo;

   guiinfo = GUIInfoCreate(img, shape, width, height, tree, filter, k, mingreylevel, numattrs, procs, dims, dls, dhs, sharedcm);
   if (guiinfo==NULL)  return(-1);
   if (GUIOpenControlWindow(guiinfo, "Control", "MT:Control"))
   {
      fprintf(stderr, "Can't open control window\n");
      GUIInfoDelete(guiinfo);
      return(-1);
   }
   if (GUIOpenDiatomWindow(guiinfo, "Diatom", "MT:Diatom"))
   {
      fprintf(stderr, "Can't open diatom window\n");
      GUIInfoDelete(guiinfo);
      return(-1);
   }
   if (GUIOpenGranWindow(guiinfo, "Granulometry", "MT:Gran"))
   {
      fprintf(stderr, "Can't open granulometry window\n");
      GUIInfoDelete(guiinfo);
      return(-1);
   }
   HandleEvents(guiinfo);
   printf("about to delete guiinfo\n");
   GUIInfoDelete(guiinfo);
   printf("guiinfo deleted\n");
   return(0);
} /* GUIShow */
