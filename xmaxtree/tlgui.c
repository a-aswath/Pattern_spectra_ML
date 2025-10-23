/* tlgui.c */
/* October 2000  Erik Urbach */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tlgui.h"



void TLRefreshStringGadget(TLGadget *gadget);




int TLAllocGrayColors(TLDisplay *display)
{
   bool buffer[256];
   int i, j, v, r;
   ulong p;

   for (i=0; i<256; i++)  buffer[i] = true;
   for (i=2; i<=256; i++)
   {
      for (j=0; j<i; j++)
      {
         v = j%i;
         r = v*255/(i-1);
         if (buffer[r])
         {
            display->Colors[r].red = r<<8;
            display->Colors[r].green = r<<8;
            display->Colors[r].blue = r<<8;
            display->Colors[r].flags = DoRed|DoGreen|DoBlue;
            if (XAllocColor(display->Display, display->ColorMap, &(display->Colors[r])))  buffer[r] = false;
            else
            {
               if (buffer[0])  return(0);
               p = display->Colors[0].pixel;
               for (i=0; i<256; i++)
               {
                  if (buffer[i])  display->Colors[i].pixel = p;
                  else  p = display->Colors[i].pixel;
               }
               return(-1);
            }
         }
      }
   }
   return(1);
} /* TLAllocGrayColors */



TLDisplay *TLCreateDisplay(bool sharedcm)
{
   TLDisplay *info;
   int i;

   info = malloc(sizeof(TLDisplay));
   if (info==NULL)  return(NULL);
   info->FirstWindow = NULL;
   info->Display = XOpenDisplay(NULL);
   if (info->Display==NULL)
   {
      free(info);
      return(NULL);
   }
   info->Screen = DefaultScreen(info->Display);
   info->Visual = DefaultVisual(info->Display, info->Screen);
   info->Depth = DefaultDepth(info->Display, info->Screen);
   info->RootWindow = RootWindow(info->Display, info->Screen);
   info->Font = XLoadQueryFont(info->Display, "8x13bold");
   if (info->Font==NULL)   fprintf(stderr, "Can't load font 8x13bold!\n");
   info->SharedColorMap = sharedcm;
   if (sharedcm)  info->ColorMap = DefaultColormap(info->Display, info->Screen);
   else
   {
      info->ColorMap = XCreateColormap(info->Display, info->RootWindow, info->Visual, AllocNone);
      if (!(info->ColorMap))
      {
         XCloseDisplay(info->Display);
         free(info);
         return(NULL);
      }
   }
   info->Red.red = 255<<8;
   info->Red.green = 0;
   info->Red.blue = 0;
   info->Red.flags = DoRed|DoGreen|DoBlue;
   if (!XAllocColor(info->Display, info->ColorMap, &(info->Red)))
   {
      printf("Can't allocate red color!\n");
      info->Red.pixel = WhitePixel(info->Display, info->Screen);
   }
   info->Orange.red = 255<<8;
   info->Orange.green = 150<<8;
   info->Orange.blue = 0;
   info->Orange.flags = DoRed|DoGreen|DoBlue;
   if (!XAllocColor(info->Display, info->ColorMap, &(info->Orange)))
   {
      printf("Can't allocate orange color!\n");
      info->Orange.pixel = info->Red.pixel;
   }
   i = TLAllocGrayColors(info);
   if (i<0)  printf("Couldn't allocate all gray colors!\n");
   else if (i==0)
   {
      printf("Not enough colors\n");
      TLDeleteDisplay(info);
      return(NULL);
   }
   info->LightPixel = info->Colors[230].pixel;
   info->DarkPixel = info->Colors[80].pixel;
   info->TextPixel = info->Colors[0].pixel;
   info->WhitePixel = WhitePixel(info->Display, info->Screen);
   return(info);
} /* TLCreateDisplay */



void TLDeleteDisplay(TLDisplay *info)
{
   TLWindow *w, *next;

   w = info->FirstWindow;
   while (w)
   {
      next = w->Next;
      TLDeleteWindow(w);
      w = next;
   }
   XFlush(info->Display);
   XCloseDisplay(info->Display);
   free(info); 
} /* TLDeleteDisplay */



int TLDisplayGetDepth(TLDisplay *info)
{
   return(info->Depth);
} /* TLDisplayGetDepth */



int TLDisplayGetDisplayCells(TLDisplay *info)
{
   return(DisplayCells(info->Display, info->Screen));
} /* TLDisplayGetDisplayCells */



int TLDisplayGetDisplayPlanes(TLDisplay *info)
{
   return(DisplayPlanes(info->Display, info->Screen));
} /* TLDisplayGetDisplayPlanes */



TLWindow *TLCreateWindow(TLDisplay *info, uint width, uint height, char *windowname, char *iconname)
{
   XSetWindowAttributes winattrs;
   TLWindow *window;
   XGCValues values;
   ulong wamask;

   window = malloc(sizeof(TLWindow));
   if (window==NULL)  return(NULL);
   window->Display = info;
   winattrs.border_pixel = BlackPixel(info->Display, info->Screen);
   winattrs.background_pixel = info->Colors[195].pixel;
   wamask = CWBackPixel | CWBorderPixel;
   if (!info->SharedColorMap)
   {
      winattrs.colormap = info->ColorMap;
      wamask |= CWColormap;
   }
   window->Window = XCreateWindow(info->Display, info->RootWindow, 0, 0, width, height, 5, info->Depth,
                                  InputOutput, info->Visual, wamask, &winattrs);
   window->FirstGadget = NULL;
   window->ActiveGadget = NULL;
   window->InputGadget = NULL;
   window->ActionFlags = 0;
   window->GC = XCreateGC(info->Display,window->Window,(unsigned long)0,&values);  
   XSetStandardProperties(info->Display, window->Window, windowname, iconname, None, 0, 0, NULL);
   XSelectInput(info->Display, window->Window,
                ButtonPressMask|ButtonReleaseMask|ExposureMask|KeyPressMask|KeyReleaseMask|LeaveWindowMask|PointerMotionMask);
   window->Next = info->FirstWindow;
   window->Prev = NULL;
   if (info->FirstWindow)  info->FirstWindow->Prev = window;
   info->FirstWindow = window;
   XSetFont(info->Display, window->GC, info->Font->fid);
   XMapWindow(info->Display, window->Window);
   return(window);
} /* TLCreateWindow */



void TLDeleteWindow(TLWindow *window)
{
   TLGadget *g, *next;

   g = window->FirstGadget;
   while (g)
   {
      next = g->Next;
      TLDeleteGadget(g);
      g = next;
   }
   XDestroyWindow(window->Display->Display, window->Window);
   if (window->Next)  window->Next->Prev = window->Prev;
   if (window->Prev)  window->Prev->Next = window->Next;
   else  window->Display->FirstWindow = window->Next;
   free(window); 
} /* TLDeleteWindow */



void TLWindowSelectInput(TLWindow *window, int actionflags)
{
   window->ActionFlags = actionflags;
} /* TLWindowSelectInput */



void TLSetForeground(TLWindow *window, ulong pixel)
{
   XSetForeground(window->Display->Display, window->GC, pixel);
} /* TLSetForeground */



void TLSetBackground(TLWindow *window, ulong pixel)
{
   XSetBackground(window->Display->Display, window->GC, pixel);
} /* TLSetBackground */



void TLDrawLine(TLWindow *window, int x1, int y1, int x2, int y2)
{
   XDrawLine(window->Display->Display, window->Window, window->GC, x1, y2, x2, y2);
} /* TLDrawLine */



void TLDrawBorder3D(TLWindow *window, int xmin, int ymin, int xmax, int ymax, int pressed)
{
   if (pressed)  TLSetForeground(window, window->Display->DarkPixel);
   else  TLSetForeground(window, window->Display->LightPixel);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmin, ymin, xmax, ymin);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmin+1, ymin+1, xmax-1, ymin+1);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmax, ymin, xmax, ymax);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmax-1, ymin+1, xmax-1, ymax-1);
   if (pressed)  TLSetForeground(window, window->Display->LightPixel);
   else  TLSetForeground(window, window->Display->DarkPixel);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmax-1, ymax, xmin, ymax);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmax-1, ymax-1, xmin+1, ymax-1);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmin, ymax, xmin, ymin+1);
   XDrawLine(window->Display->Display, window->Window, window->GC, xmin+1, ymax-1, xmin+1, ymin+2);
} /* TLDrawBorder3D */



void TLFillRectangle(TLWindow *window, int x, int y, int width, int height)
{
   XFillRectangle(window->Display->Display, window->Window, window->GC, x, y, width, height);
} /* TLFillRectangle */



int TLTextWidth(TLWindow *window, char *str, int length)
{
   return(XTextWidth(window->Display->Font, str, length));
} /* TLTextWidth */



void TLDrawString(TLWindow *window, int x, int y, char *str, int length)
{
   XDrawString(window->Display->Display, window->Window, window->GC, x, y, str, length);
} /* TLDrawString */



void TLDrawImageString(TLWindow *window, int x, int y, char *str, int length)
{
   XDrawImageString(window->Display->Display, window->Window, window->GC, x, y, str, length);
} /* TLDrawImageString */



TLImage *TLCreateImage(TLWindow *window, ubyte *data, long width, long height)
{
   TLImage *image;
   long p, x, y;

   image = malloc(sizeof(TLImage));
   if (image==NULL)  return(NULL);
   image->Data = malloc(width*height*4);
   if (image->Data==NULL)
   {
      free(image);
      return(NULL);
   }
   image->Width = width;
   image->Height = height;
   image->Image = XCreateImage(window->Display->Display, window->Display->Visual, window->Display->Depth, ZPixmap, 0, image->Data, width, height, 8, 0);
   if (image->Image==NULL)
   {
      free(image->Data);
      free(image);
      return(NULL);
   }
   for (y=0; y<height; y++)
   {
      for (x=0; x<width; x++)
      {
         p = y*width+x;
         XPutPixel(image->Image, x, y, window->Display->Colors[data[p]].pixel);
      }
   }
   return(image);
} /* TLCreateImage */



void TLRefreshImage(TLWindow *window, TLImage *image, ubyte *data)
{
   long p, x, y;

   for (y=0; y<image->Height; y++)
   {
      for (x=0; x<image->Width; x++)
      {
         p = y*(image->Width)+x;
         XPutPixel(image->Image, x, y, window->Display->Colors[data[p]].pixel);
      }
   }
} /* TLRefreshImage */



void TLDeleteImage(TLImage *image)
{
   XDestroyImage(image->Image);
   free(image);
} /* TLDeleteImage */



void TLPutImage(TLWindow *window, TLImage *image)
{
   XPutImage(window->Display->Display, window->Window, window->GC, image->Image, 0,0,0,0,image->Width,image->Height);
} /* TLPlaceImage */



TLGadget *TLCreateGadget(TLWindow *window, int type, int x, int y, int width, int height, int id, void *userdata, void *data)
{
   TLGadget *g;

   g = malloc(sizeof(TLGadget));
   if (g==NULL)  return(NULL);
   g->Window = window;
   g->Next = window->FirstGadget;
   g->Prev = NULL;
   g->Type = type;
   g->X = x;
   g->Y = y;
   g->Width = width;
   g->Height = height;
   g->ID = id;
   g->UserData = userdata;
   g->Data = data;
   if (window->FirstGadget)  window->FirstGadget->Prev = g;
   window->FirstGadget = g;
   return(g);
} /* TLCreateGadget */



TLGadget *TLCreateButton(TLWindow *window, int x, int y, int width, int height, char *contents, int id, void *userdata)
{
   TLGadget *g;
   TLButtonData *data;
   char *str;

   str = malloc(strlen(contents)+1);
   if (str==NULL)  return(NULL);
   strcpy(str, contents);
   data = malloc(sizeof(TLButtonData));
   if (data==NULL)
   {
      free(str);
      return(NULL);
   }
   data->Contents = str;
   data->Pressed = false;
   g = TLCreateGadget(window, TLGT_Button, x, y, width, height, id, userdata, data);
   if (g==NULL)
   {
      free(data);
      free(str);
      return(NULL);
   }
   return(g);
} /* TLCreateButton */



TLGadget *TLCreateStringGadget(TLWindow *window, int x, int y, int width, int height, int datatype, int maxchars,
                               int id, void *userdata)
{
   TLGadget *g;
   TLStringGadgetData *data;
   char *str;

   str = malloc(maxchars+1);
   if (str==NULL)  return(NULL);
   str[0] = 0;
   data = malloc(sizeof(TLStringGadgetData));
   if (data==NULL)
   {
      free(str);
      return(NULL);
   }
   data->Contents = str;
   data->DataType = datatype;
   data->MaxChars = maxchars;
   data->Pos = 0;
   data->Active = false;
   g = TLCreateGadget(window, TLGT_StringGadget, x, y, width, height, id, userdata, data);
   if (g==NULL)
   {
      free(data);
      free(str);
      return(NULL);
   }
   return(g);
} /* TLCreateStringGadget */



TLGadget *TLCreateTextInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata)
{
   return(TLCreateStringGadget(window, x, y, width, height, TLSGDT_Text, maxchars, id, userdata));
} /* TLCreateTextInput */



TLGadget *TLCreateIntInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata)
{
   TLGadget *gadget;

   gadget = TLCreateStringGadget(window, x, y, width, height, TLSGDT_Int, maxchars, id, userdata);
   if (gadget)  TLStringGadgetSetText(gadget, "0");
   return(gadget);
} /* TLCreateIntInput */



TLGadget *TLCreateFloatInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata)
{
   TLGadget *gadget;

   gadget = TLCreateStringGadget(window, x, y, width, height, TLSGDT_Float, maxchars, id, userdata);
   if (gadget)  TLStringGadgetSetText(gadget, "0.0");
   return(gadget);
} /* TLCreateFloatInput */



TLGadget *TLCreateLabel(TLWindow *window, int x, int y, char *label)
{
   TLGadget *g;
   char *str;

   str = malloc(strlen(label)+1);
   if (str==NULL)  return(NULL);
   strcpy(str, label);
   g = TLCreateGadget(window, TLGT_Label, x, y, -1, -1, -1, NULL, str);
   if (g==NULL)
   {
      free(str);
      return(NULL);
   }
   return(g);
} /* TLCreateLabel */



int TLStringGadgetSetPos(TLGadget *gadget, int pos)
{
   TLStringGadgetData *data;
   int l;

   data = gadget->Data;
   l = strlen(data->Contents);
   if ((pos<0) || (pos>l))  pos = l;
   data->Pos = pos;
   TLRefreshStringGadget(gadget);
   return(pos);
} /* TLStringGadgetSetPos */



int TLStringGadgetCursorLeft(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   if (data->Pos>0)
   {
      data->Pos--;
      TLRefreshStringGadget(gadget);
   }
   return(data->Pos);
} /* TLStringGadgetCursurLeft */



int TLStringGadgetCursorRight(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   if (data->Pos<strlen(data->Contents))
   {
      data->Pos++;
      TLRefreshStringGadget(gadget);
   }
   return(data->Pos);
} /* TLStringGadgetCursurRight */



int TLStringGadgetSetText(TLGadget *gadget, char *s)
{
   TLStringGadgetData *data;
   int l;

   data = gadget->Data;
   l = strlen(s);
   if (l>data->MaxChars)  l = data->MaxChars;
   strncpy(data->Contents, s, l);
   data->Contents[l] = 0;
   data->Pos = 0;
   TLRefreshStringGadget(gadget);
   return(l);
} /* TLStringGadgetSetText */



int TLIntInputSetValue(TLGadget *gadget, int value)
{
   char str[20];

   sprintf(str, "%d", value);
   return(TLStringGadgetSetText(gadget, str));
} /* TLIntInputSetValue */



int TLFloatInputSetValue(TLGadget *gadget, float value)
{
   char str[30];

   sprintf(str, "%1.1f", value);
   return(TLStringGadgetSetText(gadget, str));
} /* TLFloatInputSetValue */



char *TLStringGadgetGetText(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   return(data->Contents);
} /* TLStringGadgetGetText */



int TLIntInputGetValue(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   return(atoi(data->Contents));
} /* TLIntInputGetValue */



float TLFloatInputGetValue(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   return(atof(data->Contents));
} /* TLFloatInputGetValue */



bool TLStringGadgetInsertChar(TLGadget *gadget, char c)
{
   TLStringGadgetData *data;
   int oldlen, oldpos, i;

   data = gadget->Data;
   oldlen = strlen(data->Contents);
   oldpos = data->Pos;
   if (oldlen==data->MaxChars)  return(false);
   for (i=oldlen; i>=oldpos; i--)  data->Contents[i+1] = data->Contents[i];
   data->Contents[oldpos] = c;
   data->Pos ++;
   TLRefreshStringGadget(gadget);
   return(true);
} /* TLStringGadgetInsertChar */



int TLStringGadgetInsertText(TLGadget *gadget, char *s)
{
   TLStringGadgetData *data;
   int oldlen, oldpos, l, i;

   data = gadget->Data;
   oldlen = strlen(data->Contents);
   oldpos = data->Pos;
   l = strlen(s);
   if (oldlen+l>data->MaxChars)  l = data->MaxChars-oldlen;
   if (l<1)  return(0);
   for (i=oldlen; i>=oldpos; i--)  data->Contents[i+l] = data->Contents[i];
   for (i=0; i<l; i++)  data->Contents[oldpos+i] = s[i];
   data->Pos += l;
   TLRefreshStringGadget(gadget);
   return(l);
} /* TLStringGadgetInsertText */



bool TLStringGadgetDeleteChar(TLGadget *gadget)
{
   TLStringGadgetData *data;
   int pos, l, i;

   data = gadget->Data;
   pos = data->Pos;
   l = strlen(data->Contents);
   if (pos==l)  return(false);
   for (i=pos; i<l; i++)  data->Contents[i] = data->Contents[i+1];
   TLRefreshStringGadget(gadget);
   return(true);
} /* TLStringGadgetDeleteChar */



bool TLStringGadgetBackspaceChar(TLGadget *gadget)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   if (data->Pos)
   {
      data->Pos--;
      return(TLStringGadgetDeleteChar(gadget));
   }
   return(false);
} /* TLStringGadgetBackspaceChar */



void TLActivateStringGadget(TLGadget *gadget)
{
   TLStringGadgetData *data;

   TLDeactivateStringGadget(gadget->Window);
   gadget->Window->InputGadget = gadget;
   data = gadget->Data;
   data->Active = true;
   TLRefreshStringGadget(gadget);
} /* TLActivateStringGadget */



void TLDeactivateStringGadget(TLWindow *window)
{
   TLGadget *gadget;
   TLStringGadgetData *data;

   gadget = window->InputGadget;
   if (gadget)
   {
      window->InputGadget = NULL;
      data = gadget->Data;
      data->Active = false;
      TLRefreshStringGadget(gadget);
   }
} /* TLDeactivateStringGadget */



void TLDeleteButtonData(TLButtonData *data)
{
   free(data->Contents);
   free(data);
} /* TLDeleteButtonData */



void TLDeleteStringGadgetData(TLStringGadgetData *data)
{
   free(data->Contents);
   free(data);
} /* TLDeleteStringGadgetData */



void TLDeleteLabelData(char *data)
{
   free(data);
} /* TLDeleteLabelData */



void TLDeleteGadget(TLGadget *gadget)
{
   if (gadget->Next)  gadget->Next->Prev = gadget->Prev;
   if (gadget->Prev)  gadget->Prev->Next = gadget->Next;
   else  gadget->Window->FirstGadget = gadget->Next;
   switch(gadget->Type)
   {
      case TLGT_Button:
         TLDeleteButtonData(gadget->Data);
         break;
      case TLGT_StringGadget:
         TLDeleteStringGadgetData(gadget->Data);
         break;
      case TLGT_Label:
         TLDeleteLabelData(gadget->Data);
         break;
   }
   free(gadget);
} /* TLDeleteGadget */



TLGadget *TLFindGadgetXY(TLWindow *window, int x, int y)
{
   TLGadget *g;

   g = window->FirstGadget;
   while ((g) && ((x<g->X) || (y<g->Y) || (x>=(g->X + g->Width)) || (y>=(g->Y + g->Height))))  g = g->Next;
   return(g);
} /* TLFindGadgetXY */



void TLRefreshButton(TLGadget *gadget)
{
   TLButtonData *data;
   int x, y;

   data = gadget->Data;
   x = gadget->X;
   y = gadget->Y;
   TLDrawBorder3D(gadget->Window, x, y, x + gadget->Width-1, y + gadget->Height-1, data->Pressed);
   TLSetForeground(gadget->Window, gadget->Window->Display->TextPixel);
   TLDrawString(gadget->Window, x+5, y+15, data->Contents, strlen(data->Contents));
} /* TLRefreshButton */



void TLRefreshStringGadget(TLGadget *gadget)
{
   TLStringGadgetData *data;
   char *str;
   int x, y, textleft;

   data = gadget->Data;
   x = gadget->X;
   y = gadget->Y;
   TLDrawBorder3D(gadget->Window, x, y, x + gadget->Width-1, y + gadget->Height-1, true);
   TLSetForeground(gadget->Window, gadget->Window->Display->WhitePixel);
   TLFillRectangle(gadget->Window, x+2, y+2, gadget->Width-4, gadget->Height-4);
   TLSetForeground(gadget->Window, gadget->Window->Display->TextPixel);
   str = data->Contents;
   TLDrawString(gadget->Window, x+5, y+15, str, strlen(str));
   if (data->Active)
   {
      TLSetBackground(gadget->Window, gadget->Window->Display->Red.pixel);
      textleft = TLTextWidth(gadget->Window, str, data->Pos);
      if (data->Pos<strlen(str))  TLDrawImageString(gadget->Window, x+5+textleft, y+15, &(str[data->Pos]), 1);
      else  TLDrawImageString(gadget->Window, x+5+textleft, y+15, " ", 1);
   }
} /* TLRefreshStringGadget */



void TLRefreshLabel(TLGadget *gadget)
{
   char *str;
   int x, y;

   str = gadget->Data;
   x = gadget->X;
   y = gadget->Y;
   TLSetForeground(gadget->Window, gadget->Window->Display->TextPixel);
   TLDrawString(gadget->Window, x, y+15, str, strlen(str));
} /* TLRefreshLabel */



void TLRefreshGadget(TLGadget *gadget)
{
   switch(gadget->Type)
   {
      case TLGT_Button:
         TLRefreshButton(gadget);
         break;
      case TLGT_StringGadget:
         TLRefreshStringGadget(gadget);
         break;
      case TLGT_Label:
         TLRefreshLabel(gadget);
         break;
   }
} /* TLRefreshGadget */



void TLRefreshWindow(TLWindow *window)
{
   TLGadget *g;

   g = window->FirstGadget;
   while (g)
   {
      TLRefreshGadget(g);
      g = g->Next;
   }
} /* TLRefreshWindow */



bool TLProcessGadgetEventPress(TLGadget *gadget, int x, int y, TLEvent *event)
{
   event->Type = TLET_Gadget;
   event->GadgetEvent.Window = gadget->Window;
   event->GadgetEvent.ID = gadget->ID;
   event->GadgetEvent.UserData = gadget->UserData;
   event->GadgetEvent.X = x;
   event->GadgetEvent.Y = y;
   event->GadgetEvent.Action = TLGEA_Press;
   return(true);
} /* TLProcessGadgetEventPress */



bool TLProcessButtonEventPress(TLGadget *gadget, int x, int y, TLEvent *event)
{
   TLButtonData *data;

   data = gadget->Data;
   data->Pressed = true;
   TLRefreshButton(gadget);
   return(false);
} /* TLProcessButtonEventPress */



bool TLProcessStringGadgetEventPress(TLGadget *gadget, int x, int y, TLEvent *event)
{
   TLActivateStringGadget(gadget);
   return(false);
} /* TLProcessStringGadgetEventPress */



bool TLProcessEventGadgetPress(TLGadget *gadget, int x, int y, TLEvent *event)
{
   bool result = false;

   switch(gadget->Type)
   {
      case TLGT_Gadget:
         result = TLProcessGadgetEventPress(gadget, x, y, event);
         break;
      case TLGT_Button:
         result = TLProcessButtonEventPress(gadget, x, y, event);
         break;
      case TLGT_StringGadget:
         result = TLProcessStringGadgetEventPress(gadget, x, y, event);
         break;
   }
   return(result);
} /* TLProcessEventGadgetPress */



bool TLProcessEventMousePress(TLWindow *window, int x, int y, TLEvent *event)
{
   if (window->ActionFlags & TLWAF_MousePress)
   {
      event->Type = TLET_Mouse;
      event->MouseEvent.Window = window;
      event->MouseEvent.X = x;
      event->MouseEvent.Y = y;
      event->MouseEvent.Action = TLMEA_MousePress;
      return(true);
   }
   return(false);
} /* TLProcessEventMousePress */



bool TLProcessGadgetEventRelease(TLGadget *gadget, int x, int y, TLEvent *event)
{
   event->Type = TLET_Gadget;
   event->GadgetEvent.Window = gadget->Window;
   event->GadgetEvent.ID = gadget->ID;
   event->GadgetEvent.UserData = gadget->UserData;
   event->GadgetEvent.X = x;
   event->GadgetEvent.Y = y;
   event->GadgetEvent.Action = TLGEA_Release;
   return(true);
} /* TLProcessGadgetEventRelease */



bool TLProcessButtonEventRelease(TLGadget *gadget, int x, int y, TLEvent *event)
{
   TLButtonData *data;

   event->Type = TLET_Button;
   event->ButtonEvent.Window = gadget->Window;
   event->ButtonEvent.ID = gadget->ID;
   event->ButtonEvent.UserData = gadget->UserData;
   data = gadget->Data;
   data->Pressed = false;
   TLRefreshButton(gadget);
   return(true);
} /* TLProcessButtonEventRelease */



bool TLProcessEventGadgetRelease(TLGadget *gadget, int x, int y, TLEvent *event)
{
   bool result = false;

   switch(gadget->Type)
   {
      case TLGT_Gadget:
         result = TLProcessGadgetEventRelease(gadget, x, y, event);
         break;
      case TLGT_Button:
         result = TLProcessButtonEventRelease(gadget, x, y, event);
         break;
   }
   return(result);
} /* TLProcessEventGadgetRelease */



bool TLProcessEventMouseRelease(TLWindow *window, int x, int y, TLEvent *event)
{
   if (window->ActionFlags & TLWAF_MouseRelease)
   {
      event->Type = TLET_Mouse;
      event->MouseEvent.Window = window;
      event->MouseEvent.X = x;
      event->MouseEvent.Y = y;
      event->MouseEvent.Action = TLMEA_MouseRelease;
      return(true);
   }
   return(false);
} /* TLProcessEventMouseRelease */



bool TLProcessGadgetEventDrag(TLGadget *gadget, int x, int y, TLEvent *event)
{
   event->Type = TLET_Gadget;
   event->GadgetEvent.Window = gadget->Window;
   event->GadgetEvent.ID = gadget->ID;
   event->GadgetEvent.UserData = gadget->UserData;
   event->GadgetEvent.X = x;
   event->GadgetEvent.Y = y;
   event->GadgetEvent.Action = TLGEA_Drag;
   return(true);
} /* TLProcessGadgetEventDrag */



bool TLProcessButtonEventDrag(TLGadget *gadget, int x, int y, TLEvent *event)
{
   TLButtonData *data;
   bool p;

   if ((x<gadget->X) || (y<gadget->Y) || (x>=(gadget->X + gadget->Width)) || (y>=(gadget->Y + gadget->Height)))  p = false;
   else p = true;
   data = gadget->Data;
   if (p!=data->Pressed)
   {
      data->Pressed = p;
      TLRefreshButton(gadget);
   }
   return(false);
} /* TLProcessButtonEventDrag */



bool TLProcessEventGadgetDrag(TLGadget *gadget, int x, int y, TLEvent *event)
{
   bool result = false;

   switch(gadget->Type)
   {
      case TLGT_Gadget:
         result = TLProcessGadgetEventDrag(gadget, x, y, event);
         break;
      case TLGT_Button:
         result = TLProcessButtonEventDrag(gadget, x, y, event);
         break;
   }
   return(result);
} /* TLProcessEventGadgetDrag */



bool TLProcessEventMouseMotion(TLWindow *window, int x, int y, TLEvent *event)
{
   if (window->ActionFlags & TLWAF_MouseMove)
   {
      event->Type = TLET_Mouse;
      event->MouseEvent.Window = window;
      event->MouseEvent.X = x;
      event->MouseEvent.Y = y;
      event->MouseEvent.Action = TLMEA_MouseMove;
      return(true);
   }
   return(false);
} /* TLProcessEventMouseMotion */



TLGadget *TLStringGadgetFindNext(TLGadget *gadget)
{
   TLWindow *window;

   window = gadget->Window;
   gadget = gadget->Next;
   while ((gadget) && (gadget->Type!=TLGT_StringGadget))  gadget = gadget->Next;
   if (gadget)  return(gadget);
   gadget = window->FirstGadget;
   while (gadget->Type!=TLGT_StringGadget)  gadget = gadget->Next;
   return(gadget);
} /* TLStringGadgetFindNext */



bool TLProcessTextInputEnter(TLGadget *gadget, TLEvent *event)
{
   TLStringGadgetData *data;

   data = gadget->Data;
   event->TextInputEvent.Type = TLET_TextInput;
   event->TextInputEvent.Window = gadget->Window;
   event->TextInputEvent.ID = gadget->ID;
   event->TextInputEvent.UserData = gadget->UserData;
   event->TextInputEvent.Text = data->Contents;
   return(true);
} /* TLProcessTextInputEnter */



bool TLProcessIntInputEnter(TLGadget *gadget, TLEvent *event)
{
   TLStringGadgetData *data;

   TLIntInputSetValue(gadget, TLIntInputGetValue(gadget));
   data = gadget->Data;
   event->IntInputEvent.Type = TLET_IntInput;
   event->IntInputEvent.Window = gadget->Window;
   event->IntInputEvent.ID = gadget->ID;
   event->IntInputEvent.UserData = gadget->UserData;
   event->IntInputEvent.Value = atoi(data->Contents);
   return(true);
} /* TLProcessIntInputEnter */



bool TLProcessFloatInputEnter(TLGadget *gadget, TLEvent *event)
{
   TLStringGadgetData *data;

   TLFloatInputSetValue(gadget, TLFloatInputGetValue(gadget));
   data = gadget->Data;
   event->FloatInputEvent.Type = TLET_FloatInput;
   event->FloatInputEvent.Window = gadget->Window;
   event->FloatInputEvent.ID = gadget->ID;
   event->FloatInputEvent.UserData = gadget->UserData;
   event->FloatInputEvent.Value = atof(data->Contents);
   return(true);
} /* TLProcessFloatInputEnter */



bool TLProcessStringGadgetEnter(TLGadget *gadget, TLEvent *event)
{
   TLStringGadgetData *data;
   bool result = false;

   data = gadget->Data;
   switch(data->DataType)
   {
      case TLSGDT_Text:
         result = TLProcessTextInputEnter(gadget, event);
         break;
      case TLSGDT_Int:
         result = TLProcessIntInputEnter(gadget, event);
         break;
      case TLSGDT_Float:
         result = TLProcessFloatInputEnter(gadget, event);
         break;
   }
   return(result);
} /* TLProcessStringGadgetEnter */



bool TLProcessStringGadgetEventKeyRelease(TLGadget *gadget, char *keybuffer, int num, KeySym ks, TLEvent *event)
{
   bool result = false;

   switch(ks)
   {
      case XK_BackSpace:
         TLStringGadgetBackspaceChar(gadget);
         break;
      case XK_Tab:
      case XK_Linefeed:
      case XK_Return:
         result = TLProcessStringGadgetEnter(gadget, event);
         TLActivateStringGadget(TLStringGadgetFindNext(gadget));
         break;
      case XK_Delete:
         TLStringGadgetDeleteChar(gadget);
         break;
      case XK_Home:
         TLStringGadgetSetPos(gadget, 0);
         break;
      case XK_Left:
         TLStringGadgetCursorLeft(gadget);
         break;
      case XK_Right:
         TLStringGadgetCursorRight(gadget);
         break;
      case XK_End:
         TLStringGadgetSetPos(gadget, -1);
         break;
      default:
         TLStringGadgetInsertText(gadget, keybuffer);
         break;
   }
   return(result);
} /* TLProcessStringGadgetEventKeyRelease */



bool TLProcessEventGadgetKeyRelease(TLGadget *gadget, char *keybuffer, int num, KeySym ks, TLEvent *event)
{
   bool result = false;

   switch(gadget->Type)
   {
      case TLGT_StringGadget:
         result = TLProcessStringGadgetEventKeyRelease(gadget, keybuffer, num, ks, event);
         break;
   }
   return(result);
} /* TLProcessEventGadgetKeyRelease */



bool TLProcessEventKeyRelease(TLWindow *window, char *keybuffer, int num, KeySym ks, TLEvent *event)
{
   if (window->InputGadget)  return(TLProcessEventGadgetKeyRelease(window->InputGadget, keybuffer, num, ks, event));
   return(false);
} /* TLProcessEventKeyRelease */



bool TLProcessEventExpose(TLWindow *window, TLEvent *event)
{
   if (window->ActionFlags & TLWAF_Refresh)
   {
      event->Type = TLET_WindowRefresh;
      event->RefreshEvent.Window = window;
      return(true);
   }
   return(false);
} /* TLProcessEventExpose */



bool TLProcessEventWindowLeave(TLWindow *window, TLEvent *event)
{
   if (window->ActionFlags & TLWAF_Leave)
   {
      event->Type = TLET_WindowLeave;
      event->LeaveEvent.Window = window;
      return(true);
   }
   return(false);
} /* TLProcessEventWindowLeave */



bool TLProcessEvents(TLWindow *window, XEvent *xe, TLEvent *event)
{
   TLGadget *g;
   KeySym ks;
   char keybuffer[10];
   int x, y, num;
   bool result = false;

   switch(xe->type)
   {
      case ButtonPress:
         x = xe->xbutton.x;
         y = xe->xbutton.y;
         g = TLFindGadgetXY(window, x, y);
         if (g)  result = TLProcessEventGadgetPress(g, x, y, event);
         else  result = TLProcessEventMousePress(window, x, y, event);
         window->ActiveGadget = g;
         break;
      case ButtonRelease:
         x = xe->xbutton.x;
         y = xe->xbutton.y;
         g = TLFindGadgetXY(window, x, y);
         if ((window->ActiveGadget) && (g==window->ActiveGadget))  result = TLProcessEventGadgetRelease(g, x, y, event);
         else  result = TLProcessEventMouseRelease(window, x, y, event);
         window->ActiveGadget = NULL;
         break;
      case Expose:
         if (xe->xexpose.count==0)
         {
            TLRefreshWindow(window);
            result = TLProcessEventExpose(window, event);
         }
         break;
      case KeyPress:
         break;
      case KeyRelease:
         num = XLookupString(&(xe->xkey), keybuffer, 10, &ks, NULL);
	 result = TLProcessEventKeyRelease(window, keybuffer, num, ks, event);
         break;
      case LeaveNotify:
         result = TLProcessEventWindowLeave(window, event);
         break;
      case MotionNotify:
         x = xe->xmotion.x;
         y = xe->xmotion.y;
         if (window->ActiveGadget)  result = TLProcessEventGadgetDrag(window->ActiveGadget, x, y, event);
         else  result = TLProcessEventMouseMotion(window, x, y, event);
         break;
   }
   return(result);
} /* TLProcessEvents */



void TLEventWait(TLDisplay *display, TLEvent *event)
{
   XEvent xe;
   TLWindow *w;
   bool done = false;
 
   while(!done)
   {
      XNextEvent(display->Display, &xe);
      w = display->FirstWindow;
      while ((!done) && (w))
      {
         if (xe.xany.window==w->Window)  done = TLProcessEvents(w, &xe, event);
         if (!done)  w = w->Next;
      }
   }
} /* TLEventWait */
