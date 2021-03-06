/*
 *  Functions for sending messages to stderr and log files.
 *
 *  $Id: message.cpp,v 1.1.2.1 2006/01/13 00:24:55 jeffc Exp $
 */

#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "splitter.h"

void message(char *msg) {
  fprintf(stderr,"%s %s",tapeheaders[0].name,msg);
  fprintf(errorlog,"%s %s",tapeheaders[0].name,msg);
}



/*
 * $Log: message.cpp,v $
 * Revision 1.1.2.1  2006/01/13 00:24:55  jeffc
 * *** empty log message ***
 *
 * Revision 1.2  2003/09/11 18:53:38  korpela
 * *** empty log message ***
 *
 * Revision 1.1  2003/07/29 20:35:43  korpela
 *
 * renames .C files to .cpp
 *
 * Revision 1.1  2003/06/03 00:16:14  korpela
 *
 * Initial splitter under CVS control.
 *
 * Revision 3.0  2001/08/01 19:04:57  korpela
 * Check this in before Paul screws it up.
 *
 * Revision 2.2  2000/12/01 01:13:29  korpela
 * *** empty log message ***
 *
 * Revision 2.1  1998/11/02 16:41:21  korpela
 * Minor Change.
 *
 * Revision 2.0  1998/10/30  22:00:04  korpela
 * Conversion to C++ and merger with client source tree.
 *
 * Revision 1.1  1998/10/27  00:56:16  korpela
 * Initial revision
 *
 */
