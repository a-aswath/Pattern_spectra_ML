/* tlgui.h */
/* October 2000  Erik Urbach */

#ifndef TLGUI_H
#define TLGUI_H



#ifndef _XLIB_H_
#include <X11/Xlib.h>
#endif /* _XLIB_H */

#ifndef MTTYPES_H
#include "mttypes.h"
#endif /* MTTYPES_H */



typedef struct TLDisplay TLDisplay;
typedef struct TLWindow TLWindow;
typedef struct TLImage TLImage;
typedef struct TLGadget TLGadget;
typedef struct TLButtonData TLButtonData;
typedef struct TLStringGadgetData TLStringGadgetData;
typedef union u_TLEvent TLEvent;
typedef struct TLAnyEvent TLAnyEvent;
typedef struct TLMouseEvent TLMouseEvent;
typedef struct TLRefreshEvent TLRefreshEvent;
typedef struct TLLeaveEvent TLLeaveEvent;
typedef struct TLGadgetEvent TLGadgetEvent;
typedef struct TLButtonEvent TLButtonEvent;
typedef struct TLTextInputEvent TLTextInputEvent;
typedef struct TLIntInputEvent TLIntInputEvent;
typedef struct TLFloatInputEvent TLFloatInputEvent;



struct TLDisplay
{
   TLWindow *FirstWindow;
   Display *Display;
   int Screen;
   Visual *Visual;
   int Depth;
   Window RootWindow;
   XFontStruct *Font;
   bool SharedColorMap;
   Colormap ColorMap;
   XColor Colors[256];
   XColor Red;
   XColor Orange;
   ulong LightPixel;
   ulong DarkPixel;
   ulong TextPixel;
   ulong WhitePixel;
};

/* TLWindow ActionFlags */
#define TLWAF_MousePress    1
#define TLWAF_MouseRelease  2
#define TLWAF_MouseMove     4
#define TLWAF_Refresh       8
#define TLWAF_Leave         16

struct TLWindow
{
   TLDisplay *Display;
   TLWindow *Next;
   TLWindow *Prev;
   TLGadget *FirstGadget;
   TLGadget *ActiveGadget;
   TLGadget *InputGadget;
   int ActionFlags;
   Window Window;
   GC GC;
};

struct TLImage
{
   XImage *Image;
   char *Data;
   long Width;
   long Height;
};

/* TLGadget Types */
#define TLGT_Gadget        0
#define TLGT_Button        1
#define TLGT_StringGadget  2
#define TLGT_Label         3

struct TLGadget
{
   TLWindow *Window;
   TLGadget *Next;
   TLGadget *Prev;
   int Type;
   int X;
   int Y;
   int Width;
   int Height;
   int ID;
   void *UserData;
   void *Data;
};

struct TLButtonData
{
   char *Contents;
   bool Pressed;
};

/* TLStringGadgetData DataTypes */
#define TLSGDT_Text   0
#define TLSGDT_Int    1
#define TLSGDT_Float  2

struct TLStringGadgetData
{
   char *Contents;
   int DataType;
   int MaxChars;
   int Pos;
   bool Active;
};

struct TLAnyEvent
{
   int Type;
   TLWindow *Window;
};

/* TLMouseEvent Actions */
#define TLMEA_MousePress    0
#define TLMEA_MouseRelease  1
#define TLMEA_MouseMove     2

struct TLMouseEvent
{
   int Type;
   TLWindow *Window;
   int X;
   int Y;
   int Action;
};

struct TLRefreshEvent
{
   int Type;
   TLWindow *Window;
};

struct TLLeaveEvent
{
   int Type;
   TLWindow *Window;
};

/* TLGadgetEvent Actions */
#define TLGEA_Press    0
#define TLGEA_Release  1
#define TLGEA_Drag     2

struct TLGadgetEvent
{
   int Type;
   TLWindow *Window;
   int ID;
   void *UserData;
   int X;
   int Y;
   int Action;
};

struct TLButtonEvent
{
   int Type;
   TLWindow *Window;
   int ID;
   void *UserData;
};

struct TLTextInputEvent
{
   int Type;
   TLWindow *Window;
   int ID;
   void *UserData;
   char *Text;
};

struct TLIntInputEvent
{
   int Type;
   TLWindow *Window;
   int ID;
   void *UserData;
   int Value;
};

struct TLFloatInputEvent
{
   int Type;
   TLWindow *Window;
   int ID;
   void *UserData;
   float Value;
};

/* TLEvent Types */
#define TLET_WindowRefresh  0
#define TLET_WindowLeave    1
#define TLET_Mouse          3
#define TLET_Gadget         4
#define TLET_Button         5
#define TLET_TextInput      6
#define TLET_IntInput       7
#define TLET_FloatInput     8

union u_TLEvent
{
   int Type;
   TLAnyEvent AnyEvent;
   TLMouseEvent MouseEvent;
   TLRefreshEvent RefreshEvent;
   TLLeaveEvent LeaveEvent;
   TLGadgetEvent GadgetEvent;
   TLButtonEvent ButtonEvent;
   TLTextInputEvent TextInputEvent;
   TLIntInputEvent IntInputEvent;
   TLFloatInputEvent FloatInputEvent;
};



TLDisplay *TLCreateDisplay(bool sharedcm);
void TLDeleteDisplay(TLDisplay *info);
int TLDisplayGetDepth(TLDisplay *info);
int TLDisplayGetDisplayCells(TLDisplay *info);
int TLDisplayGetDisplayPlanes(TLDisplay *info);
TLWindow *TLCreateWindow(TLDisplay *info, uint width, uint height, char *windowname, char *iconname);
void TLDeleteWindow(TLWindow *window);
void TLWindowSelectInput(TLWindow *window, int actionflags);
void TLSetForeground(TLWindow *window, ulong pixel);
void TLSetBackground(TLWindow *window, ulong pixel);
void TLDrawLine(TLWindow *window, int x1, int y1, int x2, int y2);
void TLDrawBorder3D(TLWindow *window, int xmin, int ymin, int xmax, int ymax, int pressed);
void TLFillRectangle(TLWindow *window, int x, int y, int width, int height);
int TLTextWidth(TLWindow *window, char *str, int length);
void TLDrawString(TLWindow *window, int x, int y, char *str, int length);
void TLDrawImageString(TLWindow *window, int x, int y, char *str, int length);
TLImage *TLCreateImage(TLWindow *window, ubyte *data, long width, long height);
void TLRefreshImage(TLWindow *window, TLImage *image, ubyte *data);
void TLDeleteImage(TLImage *image);
void TLPutImage(TLWindow *window, TLImage *image);
TLGadget *TLCreateGadget(TLWindow *window, int type, int x, int y, int width, int height, int id, void *userdata, void *data);
TLGadget *TLCreateButton(TLWindow *window, int x, int y, int width, int height, char *contents, int id, void *userdata);
TLGadget *TLCreateTextInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata);
TLGadget *TLCreateIntInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata);
TLGadget *TLCreateFloatInput(TLWindow *window, int x, int y, int width, int height, int maxchars, int id, void *userdata);
TLGadget *TLCreateLabel(TLWindow *window, int x, int y, char *label);
int TLStringGadgetSetPos(TLGadget *gadget, int pos);
int TLStringGadgetCursorLeft(TLGadget *gadget);
int TLStringGadgetCursorRight(TLGadget *gadget);
int TLStringGadgetSetText(TLGadget *gadget, char *s);
int TLIntInputSetValue(TLGadget *gadget, int value);
int TLFloatInputSetValue(TLGadget *gadget, float value);
char *TLStringGadgetGetText(TLGadget *gadget);
int TLIntInputGetValue(TLGadget *gadget);
float TLFloatInputGetValue(TLGadget *gadget);
bool TLStringGadgetInsertChar(TLGadget *gadget, char c);
int TLStringGadgetInsertText(TLGadget *gadget, char *s);
bool TLStringGadgetDeleteChar(TLGadget *gadget);
bool TLStringGadgetBackspaceChar(TLGadget *gadget);
void TLActivateStringGadget(TLGadget *gadget);
void TLDeactivateStringGadget(TLWindow *window);
void TLDeleteGadget(TLGadget *gadget);
void TLEventWait(TLDisplay *display, TLEvent *event);



#endif /* TLGUI_H */
