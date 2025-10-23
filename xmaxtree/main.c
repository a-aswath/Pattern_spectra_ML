/* main.c */
/* October 2000  Erik Urbach */
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "attrs.h"
#include "gui.h"
#include "mappers.h"
#include "maxtree.h"



bool Verbose = false;
bool ExportMatlab = false;
int k = 0;


typedef struct MapperSet
{
   int (*Mapper)(double lambda, int num, double dlow, double dhigh);
} MapperSet;

MapperSet Mappers[5] =
{
   {AreaMapper},
   {LinearMapper},
   {SqrtMapper},
   {Log2Mapper},
   {Log10Mapper}
};



typedef struct AttrSet
{
   void *(*NewAuxData)(int, int);
   void (*DeleteAuxData)(void *);
   void (*AddToAuxData)(void *, int, int);
   void (*MergeAuxData)(void *, void *);
   double (*Attribute)(void *);
} AttrSet;

AttrSet Attr[19] =
{
   {NewAreaData,DeleteAreaData,AddToAreaData,MergeAreaData,AreaAttribute},
   {NewEnclRectData,DeleteEnclRectData,AddToEnclRectData,MergeEnclRectData,EnclRectAreaAttribute},
   {NewEnclRectData,DeleteEnclRectData,AddToEnclRectData,MergeEnclRectData,EnclRectDiagAttribute},
   {NewPeriData,DeletePeriData,AddToPeriData,MergePeriData,PeriAreaAttribute},
   {NewPeriData,DeletePeriData,AddToPeriData,MergePeriData,PeriPerimeterAttribute},
   {NewPeriData,DeletePeriData,AddToPeriData,MergePeriData,PeriComplexityAttribute},
   {NewPeriData,DeletePeriData,AddToPeriData,MergePeriData,PeriSimplicityAttribute},
   {NewPeriData,DeletePeriData,AddToPeriData,MergePeriData,PeriCompactnessAttribute},
   {NewInertiaData,DeleteInertiaData,AddToInertiaData,MergeInertiaData,InertiaAttribute},
   {NewInertiaData,DeleteInertiaData,AddToInertiaData,MergeInertiaData,InertiaDivA2Attribute},
   {NewJaggedData,DeleteJaggedData,AddToJaggedData,MergeJaggedData,JaggedCompactnessAttribute},
   {NewJaggedData,DeleteJaggedData,AddToJaggedData,MergeJaggedData,JaggedInertiaDivA2Attribute},
   {NewJaggedData,DeleteJaggedData,AddToJaggedData,MergeJaggedData,JaggednessAttribute},
   {NewEntropyData,DeleteEntropyData,AddToEntropyData,MergeEntropyData,EntropyAttribute},
   {NewLambdamaxData,DeleteLambdamaxData,AddToLambdamaxData,MergeLambdamaxData,LambdamaxAttribute},
   {NewPosData,DeletePosData,AddToPosData,MergePosData,PosXAttribute},
   {NewPosData,DeletePosData,AddToPosData,MergePosData,PosYAttribute},
   {NewLevelData,DeleteLevelData,AddToLevelData,MergeLevelData,LevelAttribute},
   {NewSumFluxData,DeleteSumFluxData,AddToSumFluxData,MergeSumFluxData,SumFluxAttribute}
};



void PerformFilter(MaxTree tree, int filter, int t, double lambda)
{
   if (filter==0)  MaxTreeFilterMin(tree, Attr[t].Attribute, lambda);
   else if (filter==1)  MaxTreeFilterDirect(tree, Attr[t].Attribute, lambda);
   else if (filter==2)  MaxTreeFilterMax(tree, Attr[t].Attribute, lambda);
   else if (filter==3)  MaxTreeFilterWilkinson(tree, Attr[t].Attribute, lambda);
}

void PerformMDFilter(MaxTree tree, int filter, int numattrs, ProcSet *procs, double *lambdas)
{
   if (filter==0)  MaxTreeFilterMDMin(tree, numattrs, procs, lambdas);
   else if (filter==1)  MaxTreeFilterMDDirect(tree, numattrs, procs, lambdas);
   else if (filter==2)  MaxTreeFilterMDMax(tree, numattrs, procs, lambdas);
   else if (filter==3)  MaxTreeFilterMDWilkinson(tree, numattrs, procs, lambdas);
}

void Perform1DGranulometry(MaxTree tree,
                           int filter,
                           int t,
                           int (*lambdamapper)(double lambda, int num, double dlow, double dhigh),
                           double dlow,
                           double dhigh,
                           int width,
                           FILE *mlfile,
                           char *mlname)
{
   double *result;
   double sum = 0.0;
   int i;

   result = calloc(width+1, sizeof(double));
   if (result==NULL)
   {
      fprintf(stderr, "Perform1DGranulometry: Not enough memory!\n");
      return;
   }
   if (filter==0)
   {
      MaxTreeGranulometryMin(tree, Attr[t].Attribute, lambdamapper, width, dlow, dhigh, result);
      if (Verbose)  printf("GranulometryMin:\n");
   }
   else if (filter==1)
   {
      MaxTreeGranulometryDirect(tree, Attr[t].Attribute, lambdamapper, width, dlow, dhigh, result);
      if (Verbose)  printf("GranulometryDirect:\n");
   }
   else if (filter==2)
   {
      MaxTreeGranulometryMax(tree, Attr[t].Attribute, lambdamapper, width, dlow, dhigh, result);
      if (Verbose)  printf("GranulometryMax:\n");
   }
   else if (filter==3)
   {
      MaxTreeGranulometryWilkinson(tree, Attr[t].Attribute, lambdamapper, width, dlow, dhigh, result);
      if (Verbose)  printf("GranulometryWilkinson:\n");
   }
   if (ExportMatlab)
   {
      fprintf(mlfile, "function f = %s\n\n", mlname);
      fprintf(mlfile, "clear f;\n");
      fprintf(mlfile, "f(:) = [");
      fprintf(mlfile, "%1.1f", result[0]);
      for (i=1; i<=width; i++)  fprintf(mlfile, ", %1.1f", result[i]);
      fprintf(mlfile, "];\n\n");
      fprintf(mlfile, "return;\n");
   } else {
      fprintf(mlfile, "    ");
      for (i=0; i<=width; i++)  fprintf(mlfile, "%7d ", i);
      fprintf(mlfile, "\n    ");
      for (i=0; i<=width; i++)  fprintf(mlfile, "%7.1f ", result[i]);
      fprintf(mlfile, "\nsum ");
      for (i=0; i<=width; i++)
      {
         sum += result[i];
         fprintf(mlfile, "%7.1f ", sum);
      }
      fprintf(mlfile, "\n");
   }
} /* Perform1DGranulometry */

int PerformMDGranulometry(MaxTree tree, int filter, int numattrs, ProcSet *procs, int *dims,
                          double *dls, double *dhs, FILE *mlfile, char *mlname, int k)
{
   double *result;
   long resultsize, pos = 0;
   int x, y;
   int *cnt;
   bool done = false;

   resultsize = dims[0]+1;
   for (x=1; x<numattrs; x++)  resultsize *= dims[x]+1;
   result = calloc((size_t)resultsize, sizeof(double));
   if (result==NULL)  return(-1);
   if (filter==0)
   {
      MaxTreeGranulometryMDMin(tree, numattrs, procs, dims, result, dls, dhs);
      if (Verbose)  printf("GranulometryMDMin:\n");
   }
   else if (filter==1)
   {
      MaxTreeGranulometryMDDirect(tree, numattrs, procs, dims, result, dls, dhs);
      if (Verbose)  printf("GranulometryMDDirect:\n");
   }
   else if (filter==2)
   {
      MaxTreeGranulometryMDMax(tree, numattrs, procs, dims, result, dls, dhs);
      if (Verbose)  printf("GranulometryMDMax:\n");
   }
   else if (filter==3)
   {
     MaxTreeGranulometryMDWilkinsonK(tree, numattrs, procs, dims, result, dls, dhs, k);
      if (Verbose)  printf("GranulometryMDWilkinson:\n");
   }
   cnt = calloc(numattrs, sizeof(int));
   if (cnt==NULL)
   {
      free(result);
      return(-1);
   }
   if (ExportMatlab)
   {
      fprintf(mlfile, "function f = %s\n\n", mlname);
      fprintf(mlfile, "clear f;\n");
   }
   for (x=2; x<numattrs; x++)  cnt[x] = 0;
   while (!done)
   {
      if (!ExportMatlab)
      {
         fprintf(mlfile, "Granulometry(:,:");
         for (x=2; x<numattrs; x++)  fprintf(mlfile, ",%d",cnt[x]);
         fprintf(mlfile, ")\n");
         fprintf(mlfile, "%7c ", ' ');
         for (x=0; x<=dims[0]; x++)  fprintf(mlfile, "%7d ", x);
         fprintf(mlfile, "\n");
      }
      for (y=0; y<=dims[1]; y++)
      {
         if (ExportMatlab)
         {
            fprintf(mlfile, "f(%d, :", y+1);
            for (x=2; x<numattrs; x++)  fprintf(mlfile, ", %d", cnt[x]+1);
            fprintf(mlfile, ") = [%1.1f", result[pos]);
            pos++;
            for (x=1; x<=dims[0]; x++, pos++)  fprintf(mlfile, ", %1.1f", result[pos]);
            fprintf(mlfile, "];\n");
         } else {
            fprintf(mlfile, "%7d ", y);
            for (x=0; x<=dims[0]; x++, pos++)  fprintf(mlfile, "%7.1f ", result[pos]);
            fprintf(mlfile, "\n");
	 }
      }
      fprintf(mlfile, "\n");
      cnt[2]++;
      x = 2;
      while ((x<numattrs) && (cnt[x]>dims[x]))
      {
         fprintf(mlfile, "\n");
         cnt[x] = 0;
         x++;
         if (x<numattrs)  cnt[x]++;
      }
      if (x>=numattrs)  done = true;
   }
   if (ExportMatlab)  fprintf(mlfile, "return;\n");
   free(cnt);
   free(result);
   return(0);
} /* PerformMDGranulometry */



void PrintMaxTreeMD(MaxTree t, int numattrs, ProcSet *procs)
{
   void **attrs;
   long i, idx;
   int l, a;

   for (l=0; l<NUMLEVELS; l++)
   {
      for (i=0; i<GetNumberOfNodes(l); i++)
      {
         idx = GetNumPixelsBelowLevel(l) + i;
	 printf("C^i=%ld_l=%d Area=%ld NodeStatus=%ld idx=%ld Parent=%ld -- Attrs:",i,l,t[idx].Area,t[idx].NodeStatus,idx,t[idx].Parent);
	 for (a=0; a<numattrs; a++)
         {
            attrs = t[idx].Attribute;
            printf("%f ", procs[a].Attribute(attrs[a]));
	 }
         printf("\n");
      }
   }
} /* PrintMaxTreeMD */



int *GetIntArgs(int argc, char *argv[], int pos, int *numattr)
{
   int *args;
   int i;

   for (i=pos; (i<argc) && (isdigit(*(argv[i]))); i++);
   *numattr = i-pos;
   i--;
   if (*numattr==0)  return(NULL);
   args = calloc(*numattr, sizeof(int));
   if (args==NULL)  return(NULL);
   for (; i>=pos; i--) { args[i-pos] = atoi(argv[i]);
     printf("%d ", args[i-pos]);
   }
   return(args);
} /* GetIntArgs */



double *GetDoubleArgs(int argc, char *argv[], int pos, int *numattr)
{
   double *args;
   int i;

   for (i=pos; (i<argc) && ((isdigit(*(argv[i]))) || (*(argv[i])=='-')); i++);
   *numattr = i-pos;
   i--;
   if (*numattr==0)  return(NULL);
   args = calloc(*numattr, sizeof(double));
   if (args==NULL)  return(NULL);
   for (; i>=pos; i--)  args[i-pos] = atof(argv[i]);
   return(args);
} /* GetDoubleArgs */



void PrintUsage(char *prgname)
{
      printf("Usage: %s <diatomimage> {option}\n", prgname);
      printf("Where: option = a {attr}   - Use attribute <attr1>, <attr2>, ..., <attrn>\n");
      printf("                cm         - Don't share colormap\n");
      printf("                dl {value} - Granulometry: domain - start at <value>\n");
      printf("                dh {value} - Granulometry: domain - end at <value>\n");
      printf("                e <name>   - Write granulometry as a matlab function <name> to <name>.m\n");
      printf("                f <filter> - Use decision <filter>\n");
      printf("                g {value} -  grey threshold {value}\n");
      printf("                k {value} -  k-flat filtering value {value}\n");
      printf("                l {value}  - Filter image with lambda1=<value1>, ...\n");
      printf("                m {mapper} - Granulometry: use lambdamappers <mapper1>...<mappern>\n");
      printf("                n {size}   - Granulometry: size <n1>x<n2>x...x<nn>\n");
      printf("                nogui      - Don't pop up GUI\n");
      printf("                o <name>   - Write filtered image to <name>\n");
      printf("                s <name>   - Use shape file <name>\n");
      printf("                v          - Verbose\n");
      printf("       attr = 0 - Area (default)\n");
      printf("              1 - Area of the minimum enclosing rectangle\n");
      printf("              2 - Length of the diagonal of the minimum encl. rect.\n");
      printf("              3 - Area (Peri)\n");
      printf("              4 - Perimeter (Peri)\n");
      printf("              5 - Complexity (Peri)\n");
      printf("              6 - Simplicity (Peri)\n");
      printf("              7 - Compactness (Peri)\n");
      printf("              8 - Moment Of Inertia\n");
      printf("              9 - (Moment Of Inertia) / (Area*Area)\n");
      printf("             10 - Compactnes                          (Jagged)\n");
      printf("             11 - (Moment Of Inertia) / (Area*Area)   (Jagged)\n");
      printf("             12 - Jaggedness                          (Jagged)\n");
      printf("             13 - Entropy\n");
      printf("             14 - Lambda-Max (not idempotent -> not a filter)\n");
      printf("             15 - Max. Pos. X\n");
      printf("             16 - Max. Pos. Y\n");
      printf("             17 - Grey level\n");
      printf("             18 - Sum grey levels\n");
      printf("       filter = 0 - \"Min\" decision\n");
      printf("                1 - \"Direct\" decision (default)\n");
      printf("                2 - \"Max\" decision\n");
      printf("                3 - Wilkinson decision\n");
      printf("       mapper = 0 - Area mapper\n");
      printf("                1 - Linear mapper\n");
      printf("                2 - Sqrt mapper\n");
      printf("                3 - Log2 mapper\n");
      printf("                4 - Log10 mapper\n");
} /* PrintUsage */


typedef struct CmdlineArgs CmdlineArgs;

struct CmdlineArgs
{
   char *diatomfname;
   char *outfname;
   char *shapefname;
   char *mlname;
   double *dls;
   double *dhs;
   double *lambdas;
   int *attrs;
   int *mappers;
   int *dims;
   int numattrs;
   int numdls;
   int numdhs;
   int numlambdas;
   int nummappers;
   int numdims;
   int filter;
   int k;
   int mingreylevel;
   bool do_gran;
   bool do_filter;
   bool do_gui;
   bool SharedColormap;
   bool Verbose;
   bool ExportMatlab;
};



CmdlineArgs *CreateCmdlineArgs(void)
{
   CmdlineArgs *cmdargs;

   cmdargs = malloc(sizeof(CmdlineArgs));
   if (cmdargs==NULL)  return(NULL);
   cmdargs->diatomfname = NULL;
   cmdargs->outfname = NULL;
   cmdargs->shapefname = NULL;
   cmdargs->mlname = NULL;
   cmdargs->dls = NULL;
   cmdargs->dhs = NULL;
   cmdargs->lambdas = NULL;
   cmdargs->attrs = NULL;
   cmdargs->mappers = NULL;
   cmdargs->dims = NULL;
   cmdargs->numattrs = 0;
   cmdargs->numdls = 0;
   cmdargs->numdhs = 0;
   cmdargs->numlambdas = 0;
   cmdargs->nummappers = 0;
   cmdargs->numdims = 0;
   cmdargs->do_gran = false;
   cmdargs->do_filter = false;
   cmdargs->do_gui = true;
   cmdargs->SharedColormap = true;
   cmdargs->Verbose = false;
   cmdargs->ExportMatlab = false;
   cmdargs->k = 0;
   cmdargs->mingreylevel = 0;
   return(cmdargs);
} /* CreateCmdlineArgs */



void DeleteCmdlineArgs(CmdlineArgs *cmdargs)
{
   if (cmdargs->dims)  free(cmdargs->dims);
   if (cmdargs->mappers)  free(cmdargs->mappers);
   if (cmdargs->attrs)  free(cmdargs->attrs);
   if (cmdargs->lambdas)  free(cmdargs->lambdas);
   if (cmdargs->dhs)  free(cmdargs->dhs);
   if (cmdargs->dls)  free(cmdargs->dls);
   free(cmdargs);
} /* DeleteCmdlineArgs */



char *CmdCreateString(char *s)
{
   char *new;

   new = malloc(strlen(s)+1);
   if (new==NULL)  return(NULL);
   strcpy(new, s);
   return(new);
} /* CmdCreateString */



int *CmdCreateIntArray(int a, int b)
{
   int *new;

   new = calloc(2, sizeof(int));
   if (new==NULL)  return(NULL);
   new[0] = a;
   new[1] = b;
   return(new);
} /* CmdCreateIntArray */



double *CmdCreateDoubleArray(double a, double b)
{
   double *new;

   new = calloc(2, sizeof(double));
   if (new==NULL)  return(NULL);
   new[0] = a;
   new[1] = b;
   return(new);
} /* CmdCreateDoubleArray */



void InitCmdlineArgs(CmdlineArgs *cmdargs)
{
   cmdargs->outfname = "out.pgm";
} /* InitCmdlineArgs */



CmdlineArgs *ParseCommandlineArgs(int argc, char *argv[])
{
   CmdlineArgs *cmdargs;
   int i;

   cmdargs = CreateCmdlineArgs();
   if (cmdargs==NULL)  return(NULL);
   InitCmdlineArgs(cmdargs);
   if (argc<2)  return(cmdargs);
   cmdargs->diatomfname = argv[1];
   for (i=2; i<argc; i++)
   {
      if (strcmp(argv[i],"a")==0)
      {
         if (cmdargs->attrs)  free(cmdargs->attrs);
	 printf("Attributes:");
         cmdargs->attrs = GetIntArgs(argc, argv, i+1, &(cmdargs->numattrs));
	 if (cmdargs->attrs==NULL)
	 {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
	 }
         i += cmdargs->numattrs;
	 printf("\n");
      } else if (strcmp(argv[i],"cm")==0)
      {
         cmdargs->SharedColormap = false;
      } else if (strcmp(argv[i],"dh")==0)
      {
         if (cmdargs->dhs)  free(cmdargs->dhs);
         cmdargs->dhs = GetDoubleArgs(argc, argv, i+1, &(cmdargs->numdhs));
	 if (cmdargs->dhs==NULL)
	 {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
	 }
         i += cmdargs->numdhs;
	 printf("Num dhs = %d\n",cmdargs->numdhs);
	 cmdargs->do_gran = true;
      } else if (strcmp(argv[i],"dl")==0)
      {
         if (cmdargs->dls)  free(cmdargs->dls);
         cmdargs->dls = GetDoubleArgs(argc, argv, i+1, &(cmdargs->numdls));
	 if (cmdargs->dls==NULL)
	 {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
	 }
         i += cmdargs->numdls;
	 printf("Num dls = %d\n",cmdargs->numdls);
	 cmdargs->do_gran = true;
      } else if (strcmp(argv[i],"e")==0)
      {
         i++;
         cmdargs->mlname = argv[i];
         cmdargs->ExportMatlab = true;
      } else if (strcmp(argv[i],"f")==0)
      {
         i++;
         cmdargs->filter = atoi(argv[i]);
      }  else if (strcmp(argv[i],"g")==0)
      {
         i++;
         cmdargs->mingreylevel = atoi(argv[i]);
      }  else if (strcmp(argv[i],"k")==0)
      {
         i++;
         cmdargs->k = atoi(argv[i]);
      } else if (strcmp(argv[i],"l")==0)
      {
         if (cmdargs->lambdas)  free(cmdargs->lambdas);
         cmdargs->lambdas = GetDoubleArgs(argc, argv, i+1, &(cmdargs->numlambdas));
	 if (cmdargs->lambdas==NULL)
	 {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
	 }
         i += cmdargs->numlambdas;
	 cmdargs->do_filter = true;
      } else if (strcmp(argv[i],"m")==0)
      {
         if (cmdargs->mappers)  free(cmdargs->mappers);
         cmdargs->mappers = GetIntArgs(argc, argv, i+1, &(cmdargs->nummappers));
	 if (cmdargs->mappers==NULL)
         {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
         }
         i += cmdargs->nummappers;
	 cmdargs->do_gran = true;
	 printf("Num mappers = %d\n",cmdargs->nummappers);
      } else if (strcmp(argv[i],"n")==0)
      {
         if (cmdargs->dims)  free(cmdargs->dims);
         cmdargs->dims = GetIntArgs(argc, argv, i+1, &(cmdargs->numdims));
	 if (cmdargs->dims==NULL)
         {
            DeleteCmdlineArgs(cmdargs);
            return(NULL);
         }
         i += cmdargs->numdims;
	 printf("Num dims = %d\n",cmdargs->numdims);
	 cmdargs->do_gran = true;
      } else if (strcmp(argv[i],"nogui")==0)
      {
         cmdargs->do_gui = false;
      } else if (strcmp(argv[i],"o")==0)
      {
         i++;
         cmdargs->outfname = argv[i];
      } else if (strcmp(argv[i],"s")==0)
      {
         i++;
         cmdargs->shapefname = argv[i];
      } else if (strcmp(argv[i],"v")==0)
      {
         cmdargs->Verbose = true;
      } else {
         fprintf(stderr, "Bad option '%s'\n", argv[i]);
         DeleteCmdlineArgs(cmdargs);
         return(NULL);
      }
   }
   return(cmdargs);
} /* ParseCommandlineArgs */



void PrintCmdArgs(CmdlineArgs *cmdargs)
{
   int i;

   printf("Commandline arguments:\n");
   printf("  diatomfname='%s'\n", cmdargs->diatomfname);
   printf("  a:attrs:");
   for (i=0; i<cmdargs->numattrs; i++)  printf(" %d", cmdargs->attrs[i]);
   printf("\n");
   printf("  cm:Use shared colormap=%d\n", cmdargs->SharedColormap);
   printf("  dls&dhs:domain:");
   if (cmdargs->do_gran)
   {
      for (i=0; i<cmdargs->numattrs; i++)  printf(" %f..%f", cmdargs->dls[i], cmdargs->dhs[i]);
      printf("\n");
   } else  printf("No granulometry!\n");
   printf("  e:ExportMatlab=%d  mlname=", cmdargs->ExportMatlab);
   if (cmdargs->mlname==NULL)  printf("NULL");
   else  printf("'%s'", cmdargs->mlname);
   printf("\n");
   printf("  f:filter: decision=%d\n", cmdargs->filter);
   printf("  g:min greylevel: =%d\n", cmdargs->mingreylevel);
   printf("  k:contrast: =%d\n", cmdargs->k);
   printf("  l:lambdas:");
   if (cmdargs->do_filter)
   {
      for (i=0; i<cmdargs->numlambdas; i++)  printf(" %f", cmdargs->lambdas[i]);
      printf("\n");
   } else  printf("No filter!\n");
   printf("  m:");
   if (cmdargs->do_gran)
   {
      for (i=0; i<cmdargs->nummappers; i++)  printf(" %d", cmdargs->mappers[i]);
      printf("\n");
   } else  printf("No granulometry!\n");
   printf("  n:");
   if (cmdargs->do_gran)
   {
      for (i=0; i<cmdargs->numdims; i++)  printf(" %d", cmdargs->dims[i]);
      printf("\n");
   } else  printf("No granulometry!\n");
   printf("  do_gui:%d\n", cmdargs->do_gui);
   printf("  o:outfname='%s'\n", cmdargs->outfname);
   if (cmdargs->shapefname)  printf("  s:shapefname='%s'\n", cmdargs->shapefname);
   else  printf("  s:shapefname=NULL (not specified; using white image instead)\n");
   printf("  v:Verbose=%d\n", cmdargs->Verbose);
} /* PrintCmdArgs */


int CheckCmdlineArgs(CmdlineArgs *cmdargs)
{
   int numattrs, i;
   bool do_gran;

   numattrs = cmdargs->numattrs;
   do_gran = cmdargs->do_gran;

   if (do_gran && ((numattrs!=cmdargs->nummappers)||(numattrs!=cmdargs->numdims)||(numattrs!=cmdargs->numdhs)||(numattrs!=cmdargs->numdls)))
   {
      fprintf(stderr, "Attributes, mappers, dimensions and min/max parameters should have an equal number of values!\n");
      return(-1);
   }
   else if (cmdargs->do_filter && (numattrs!=cmdargs->numlambdas))
   {
      fprintf(stderr, "Attributes and lambdas should have an equal number of values!\n");
      return(-1);
   }
   else if ((cmdargs->filter<0) || (cmdargs->filter>3))
   {
      fprintf(stderr, "Unknown filter %d!\n", cmdargs->filter);
      return(-1);
   }
   for (i=0; i<numattrs; i++)
   {
      if ((cmdargs->attrs[i]<0) || (cmdargs->attrs[i]>18))
      {
         fprintf(stderr, "Unknown attribute %d!\n", cmdargs->attrs[i]);
         return(-1);
      }
      else if (do_gran && ((cmdargs->mappers[i]<0) || (cmdargs->mappers[i]>4)))
      {
         fprintf(stderr, "Unknown mapper %d!\n", cmdargs->mappers[i]);
         return(-1);
      }
      else if (do_gran && (cmdargs->dims[i]<=0))
      {
         fprintf(stderr, "Granulometry dimension size %d is too small!\n", cmdargs->dims[i]);
         return(-1);
      }
   }
   return(0);
} /* CheckCmdlineArgs */



int main(int argc, char *argv[])
{
   MaxTree tree;
   ProcSet *procs;
   FILE *mlfile = stdout;
   char *mlfname = NULL;
   ubyte *diatompgm, *shapepgm;
   double lamb = 6.0, dmin=1.0, dmax=0.0;
   long diatomwidth, diatomheight;
   int i, t=0, width=8, lambdamapper=0;

   CmdlineArgs *cmdargs;

   if (argc<2)
   {
      PrintUsage(argv[0]);
      return(0);
   }
   cmdargs = ParseCommandlineArgs(argc, argv);
   if (cmdargs==NULL)  return(-1);
   if (CheckCmdlineArgs(cmdargs))
   {
      DeleteCmdlineArgs(cmdargs);
      return(-1);
   }
   if (cmdargs->numattrs==1)
   {
      t = cmdargs->attrs[0];
      if (cmdargs->do_gran)
      {
         dmin = cmdargs->dls[0];
         dmax = cmdargs->dhs[0];
         width = cmdargs->dims[0];
         lambdamapper = cmdargs->mappers[0];
      }
      if (cmdargs->do_filter)  lamb = cmdargs->lambdas[0];
   }
   if (cmdargs->mlname)
   {
      mlfname = malloc(strlen(cmdargs->mlname)+3);
      if (mlfname)  sprintf(mlfname, "%s.m", cmdargs->mlname);
   }
   if (cmdargs->Verbose)  PrintCmdArgs(cmdargs);
   ReadDiatom(cmdargs->diatomfname, cmdargs->shapefname, &diatompgm, &shapepgm, &diatomwidth, &diatomheight);
   if (mlfname)
   {
      mlfile = fopen(mlfname, "w");
      if (mlfile==NULL)
      {
         fprintf(stderr, "Can't create matlab file '%s'!\nWriting to stdout\n", mlfname);
         mlfile = stdout;
      }
   }
   if (cmdargs->numattrs <= 1)
   {
      tree = MaxTreeBuild(Attr[t].NewAuxData, Attr[t].AddToAuxData, Attr[t].MergeAuxData);
      if (cmdargs->do_gran)  Perform1DGranulometry(tree, cmdargs->filter, t, Mappers[lambdamapper].Mapper, dmin, dmax, width, mlfile, cmdargs->mlname);
      if (cmdargs->do_filter)  PerformFilter(tree, cmdargs->filter, t, lamb);
      MaxTreeDestroy(Attr[t].DeleteAuxData);
   } else {
      procs = calloc(cmdargs->numattrs, sizeof(ProcSet));
      if (procs==NULL)
      {
         fprintf(stderr, "Not enough memory!\n");
         return(-1);
      }
      for (i=0; i<cmdargs->numattrs; i++)
      {
	/*printf("Adding attr %d as attribute %d\n",cmdargs->attrs[i],i);*/
         procs[i].NewAuxData = Attr[cmdargs->attrs[i]].NewAuxData;
         procs[i].DeleteAuxData = Attr[cmdargs->attrs[i]].DeleteAuxData;
         procs[i].AddToAuxData = Attr[cmdargs->attrs[i]].AddToAuxData;
         procs[i].MergeAuxData = Attr[cmdargs->attrs[i]].MergeAuxData;
         procs[i].Attribute = Attr[cmdargs->attrs[i]].Attribute;
         if (cmdargs->do_gran){
	   procs[i].Mapper = Mappers[cmdargs->mappers[i]].Mapper;
	  /* printf("Adding mapper %d to attribute %d\n",cmdargs->mappers[i],i);	  */ 
	 }
      }
      k = cmdargs->k;
   /*   printf("Starting Max-Tree build\n");*/
      tree = MaxTreeBuildMD(cmdargs->numattrs, procs);
   /*  printf("MaxTreeBuildMD done\n");*/
      if (tree==NULL)  fprintf(stderr, "Not enough memory to build MaxTree!\n");
      else
      {
         if (cmdargs->Verbose)  PrintMaxTreeMD(tree, cmdargs->numattrs, procs);
         if (cmdargs->do_gui) {
	   printf("starting GUI\n");
      fprintf(stderr, "Output filename: %s\n", cmdargs->outfname);
      mlfname = malloc(strlen(cmdargs->outfname) + 5);  
      if (mlfname)
         sprintf(mlfname, "%s.ppm", cmdargs->outfname);
	   GUIShow(diatompgm, shapepgm, diatomwidth, diatomheight, tree, cmdargs->filter,
		   cmdargs->k, cmdargs->mingreylevel, cmdargs->numattrs, procs,
		   cmdargs->dims, cmdargs->dls, cmdargs->dhs, cmdargs->SharedColormap);
	   printf("Reached GUI end\n");
	 }
         else if (cmdargs->do_gran)  PerformMDGranulometry(tree, cmdargs->filter, cmdargs->numattrs, procs, cmdargs->dims, cmdargs->dls, cmdargs->dhs, mlfile, cmdargs->mlname,cmdargs->k);
         if (cmdargs->do_filter)  PerformMDFilter(tree, cmdargs->filter, cmdargs->numattrs, procs, cmdargs->lambdas);
         MaxTreeDestroyMD(cmdargs->numattrs, procs);
	 /*printf("Max-tree removed\n");*/
      }
   }

   if (mlfile!=stdout)  fclose(mlfile);
   if (mlfname)  free(mlfname);

   WritePGM(cmdargs->outfname, diatomwidth, diatomheight);
   free(shapepgm);
   free(diatompgm);
   DeleteCmdlineArgs(cmdargs);
   return(0);
} /* main */
