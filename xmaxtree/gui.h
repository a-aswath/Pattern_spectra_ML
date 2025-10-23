/* gui.h */
/* October 2000  Erik Urbach */

#ifndef GUI_H
#define GUI_H



#ifndef MAXTREE_H
#include "maxtree.h"
#endif /* MAXTREE_H */

#ifndef MTTYPES_H
#include "mttypes.h"
#endif /* MTTYPES_H */



int GUIShow(ubyte *img, ubyte *shape, long width, long height, MaxTree tree, int filter, int k, int mingreylevel, int numattrs,
            ProcSet *procs, int *dims, double *dls, double *dhs, bool sharedcm);



#endif /* GUI_H */
