/* maxtree.h */
/* October 2000  Erik Urbach */

#ifndef MAXTREE_H
#define MAXTREE_H



#ifndef MTTYPES_H
#include "mttypes.h"
#endif /* MTTYPES_H */



#define NUMLEVELS     256
#define CONNECTIVITY  4



/*** Max-Tree definition ***/

/* MaxNode Status */
#define MTS_Ok         0
#define MTS_Deleting   1
#define MTS_Deleted    2
#define MTS_Processed  3
#define MTS_ADDED      4

struct MaxNode
{
   long Parent;
   long Area;
   void *Attribute;
   int *Pos;
   int Level;
   int PeakLevel;
   long NodeStatus;
};

typedef struct MaxNode MaxNode;
typedef struct MaxNode *MaxTree;



typedef struct ProcSet
{
   void *(*NewAuxData)(int, int);
   void (*DeleteAuxData)(void *);
   void (*AddToAuxData)(void *, int, int);
   void (*MergeAuxData)(void *, void *);
   double (*Attribute)(void *);
   int (*Mapper)(double lambda, int num, double dlow, double dhigh);
} ProcSet;



long GetNumberOfNodes(int level);
long GetNumPixelsBelowLevel(int level);
long GetNodeIndex(long p, int h);

ubyte *ReadPGM(char *fname, long *width, long *height);
void WritePGM(char *fname, long width, long height);
void ReadDiatom(char *diatomname, char *shapename, ubyte **diatomimage, ubyte **shapeimage, long *width, long *height);
MaxTree MaxTreeCreate(long size);
int GetNeighbors(long p, long *neighbors);
int flood(int h,
          void *(*NewAuxData)(int, int),
          void (*AddToAuxData)(void *, int, int),
          void (*MergeAuxData)(void *, void *),
          long *thisarea,
          int  *thispeaklevel,
          void **thisattr);
int floodMD(int h, int numattrs, ProcSet *procs, long *thisarea, int  *thispeaklevel, void **thisattrs);
void MaxTreeFilterMin(MaxTree t, double (*Attribute)(void *), double lambda);
void MaxTreeFilterMDMin(MaxTree t, int numattrs, ProcSet *procs, double *lambdas);
void MaxTreeGranulometryMin(MaxTree t,
                            double (*Attribute)(void *),
                            int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                            int num,    /* number of result entries */
                            double dlow,
                            double dhigh,
                            double *result);
void MaxTreeGranulometryMDMin(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result, double *dls, double *dhs);
void MaxTreeFilterDirect(MaxTree t, double (*Attribute)(void *), double lambda);
void MaxTreeFilterMDDirect(MaxTree t, int numattrs, ProcSet *procs, double *lambdas);
void MaxTreeGranulometryDirect(MaxTree t,
                               double (*Attribute)(void *),
                               int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                               int num,    /* number of result entries */
                               double dlow,
                               double dhigh,
                               double *result);
void MaxTreeGranulometryMDDirect(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result, double *dls, double *dhs);
void MaxTreeFilterMax(MaxTree t, double (*Attribute)(void *), double lambda);
void MaxTreeFilterMDMax(MaxTree t, int numattrs, ProcSet *procs, double *lambdas);
void MaxTreeGranulometryMax(MaxTree t,
                            double (*Attribute)(void *),
                            int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                            int num,    /* number of result entries */
                            double dlow,
                            double dhigh,
                            double *result);
void MaxTreeGranulometryMDMax(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result, double *dls, double *dhs);
void MaxTreeFilterWilkinson(MaxTree t, double (*Attribute)(void *), double lambda);
void MaxTreeFilterMDWilkinson(MaxTree t, int numattrs, ProcSet *procs, double *lambdas);
void MaxTreeGranulometryWilkinson(MaxTree t,
                                  double (*Attribute)(void *),
                                  int (*LambdaMap)(double lambda, int num, double dlow, double dhigh),
                                  int num,    /* number of result entries */
                                  double dlow,
                                  double dhigh,
                                  double *result);
void MaxTreeGranulometryMDWilkinson(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
                                    double *dls, double *dhs);
void MaxTreeGranulometryMDWilkinsonK(MaxTree t, int numattrs, ProcSet *procs, int *dims, double *result,
				     double *dls, double *dhs, int k);
MaxTree MaxTreeBuild(void *(*NewAuxData)(int, int),
                     void (*AddToAuxData)(void *, int, int),
                     void (*MergeAuxData)(void *, void *));
void MaxTreeDestroy(void (*DeleteAuxData)(void *));
MaxTree MaxTreeBuildMD(int num, ProcSet *procs);
void MaxTreeDestroyMD(int num, ProcSet *procs);



#endif /* MAXTREE_H */
