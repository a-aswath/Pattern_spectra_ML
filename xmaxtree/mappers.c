/* mappers.c */
/* October 2000  Erik Urbach */
#include <math.h>
#include "mappers.h"



int AreaMapper(double lambda, int num, double dlow, double dhigh)
/* TODO: Implement dlow and dhigh */
{
   double l = 1.0;
   int i;
 
   for (i=0; (i<num) && (lambda>=l); i++)  l *= 2.0;
   return(i);
} /* AreaMapper */



int LinearMapper(double lambda, int num, double dlow, double dhigh)
{
   int l;

   if (dlow<dhigh)  l = (lambda-dlow)*num/(dhigh-dlow);
   else  l = lambda;
   if (l<0) return(0);
   if (l>num)  return(num);
   else  return(l);
} /* LinearMapper */



int SqrtMapper(double lambda, int num, double dlow, double dhigh)
/* PRE: lambda, dlow and dhigh >= 0
 *      num > 0
 *      dlow != dhigh
 * NOTE: if dlow>dhigh then dlow and dhigh are ignored.
 */ 
{
   double r, l, h;

   r = sqrt(lambda);
   l = sqrt(dlow);
   h = sqrt(dhigh);
   if (l<h)  r = (r-l)*num/(h-l);
   if (r<0.0)  r = 0.0;
   if (r>num)  r = num;
   return((int)r);
} /* SqrtMapper */



int Log2Mapper(double lambda, int num, double dlow, double dhigh)
/* PRE: lambda, num, dlow and dhigh > 0
 *      dlow != dhigh
 * NOTE: if dlow>dhigh then dlow and dhigh are ignored.
 */ 
{
   double r, l, h;

   r = log(lambda)/log(2.0);
   l = log(dlow)/log(2.0);
   h = log(dhigh)/log(2.0);
   if (l<h)  r = (r-l)*num/(h-l);
   if (r<0.0) r = 0.0;
   if (r>num)  r = num;
   return((int)r);
} /* Log2Mapper */



int Log10Mapper(double lambda, int num, double dlow, double dhigh)
/* PRE: lambda, num, dlow and dhigh > 0
 *      dlow != dhigh
 * NOTE: if dlow>dhigh then dlow and dhigh are ignored.
 */ 
{
   double r, l, h;

   r = log(lambda)/log(10.0);
   l = log(dlow)/log(10.0);
   h = log(dhigh)/log(10.0);
   if (l<h)  r = (r-l)*num/(h-l);
   if (r<0.0) r = 0.0;
   if (r>num)  r = num;
   return((int)r);
} /* Log10Mapper */
