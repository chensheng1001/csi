/*
 * (c) 2008-2011 Daniel Halperin <dhalperi@cs.washington.edu>
 */
#include "iwl_connector.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <linux/netlink.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/kd.h>


//---------------------------------------------
/**
 * Modified by WiX 
 * @DATE 2015.11.17
 */ 

#include <memory.h>
#include <sys/time.h>
#include <time.h>

#define MAX_DATABLOCK_COUNT 5000
#define MAX_PAYLOAD 2048
#define SLOW_MSG_CNT 1
#define OUTPUT_GAP 1000

void close_file();
//----------------------------------------------

int sock_fd = -1;	// the socket
FILE* out = NULL;

/* [BEGIN] Modified by smallant 2018.1.24*/
FILE* timeOut = NULL;
/* [END] Modified by smallant */
void check_usage(int argc, char** argv);

FILE* open_file(char* filename, char* spec);

void caught_signal(int sig);

void exit_program(int code);
void exit_program_err(int code, char* func);

int main(int argc, char** argv)
{
    /* Local variables */
	struct sockaddr_nl proc_addr, kern_addr;// addrs for recv, send, bind
	struct cn_msg *cmsg;
	char buf[512*4];
	int ret;
	unsigned short l, l2;

	/* Make sure usage is correct */
	check_usage(argc, argv);

    /**
     * Base File Path..  
     *
     * @author WiX
     * @date 2015.11.17
     */ 
    char *baseFilePath = argv[1];
	
    int start_file_cnt = atoi(argv[2]);
    int FILE_CNT = atoi(argv[3]);
	int file_cnt = start_file_cnt;

    char filePath[128];
    memset(filePath, 0, sizeof(filePath));
    sprintf(filePath, "%s%d0.dat", baseFilePath, file_cnt);
	out = open_file(filePath, "w");

    /* [BEGIN] Modified by smallant 2018.1.24 */
    char timeFilePath[128];
    memset(timeFilePath, 0, sizeof(timeFilePath));
    sprintf(timeFilePath, "%s%d0-time.txt", baseFilePath, file_cnt);
    timeOut = open_file(timeFilePath, "w");
    /* [END] Modefied by smallant */

	/* Setup the socket */
	sock_fd = socket(PF_NETLINK, SOCK_DGRAM, NETLINK_CONNECTOR);
	if (sock_fd == -1)
		exit_program_err(-1, "socket");

	long recvbuf_size = 2*1024*1024;
	//if(setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &recvbuf_size, sizeof(long)) < 0)
	//SO_RCVBUF: the maximal size can be 320*1024
	//SO_RCVBUFFORCE: the maximal size can be 4*1024*1024
 	if(setsockopt(sock_fd, SOL_SOCKET, SO_RCVBUFFORCE, (const char*)&recvbuf_size, sizeof(recvbuf_size)) < 0)
	{
		exit_program_err(-1, "setsockopt");
	}

	/* Initialize the address structs */
	memset(&proc_addr, 0, sizeof(struct sockaddr_nl));
	proc_addr.nl_family = AF_NETLINK;
	proc_addr.nl_pid = getpid();			// this process' PID
	proc_addr.nl_groups = CN_IDX_IWLAGN;
	memset(&kern_addr, 0, sizeof(struct sockaddr_nl));
	kern_addr.nl_family = AF_NETLINK;
	kern_addr.nl_pid = 0;					// kernel
	kern_addr.nl_groups = CN_IDX_IWLAGN;

	/* Now bind the socket */
	if (bind(sock_fd, (struct sockaddr *)&proc_addr, sizeof(struct sockaddr_nl)) == -1)
	{	
		exit_program_err(-1, "bind");
	}

	/* And subscribe to netlink group */
	int on = proc_addr.nl_groups;
	ret = setsockopt(sock_fd, 270, NETLINK_ADD_MEMBERSHIP, &on, sizeof(on));//define SOL_NETLINK 270, kernel netlink
	if (ret)
	{
		exit_program_err(-1, "setsockopt");
	}
	/* Set up the "caught_signal" function as this program's sig handler */
	signal(SIGINT, caught_signal);

	struct timeval startTime, endTime;
        /* [BEGIN] Modified by smallant 2018.1.24 */
	struct timeval currentTime;
        /* [END] Modefied by smallant */
	gettimeofday(&startTime, 0);
    printf("INFO: Sock is ready..\n");

    int sndfd = open("/dev/console", O_WRONLY);
    //printf("sndfd = %d\n", sndfd);

    if(sndfd == -1)
    {
        perror("ERROR:Could not open sound fd\n");
        exit(0);
    }
	
	/* Poll socket forever waiting for a message */
	int count = 1;
	while (1)
	{
		/* Receive from socket with infinite timeout */
		ret = recv(sock_fd, buf, sizeof(buf), 0);
		if (ret == -1)
		{
			//exit_program_err(-1, "recv");
			perror("ERROR:[RECV]");
			memset(buf, 0, sizeof(buf));
			continue;				
		}	
		/* Pull out the message portion and print some stats */
		cmsg = NLMSG_DATA(buf);

		/* Log the data to file */
		l = (unsigned short) cmsg->len;
		l2 = htons(l);
		fwrite(&l2, 1, sizeof(unsigned short), out);
		ret = fwrite(cmsg->data, 1, l, out);

		/* [BEGIN] Modified by smallant 2018.1.24 */
		/* Log the time info to file */
		gettimeofday(&currentTime, 0);
		/* double timeInfo = currentTime.tv_sec * 1000000 + currentTime.tv_usec ; */

		/* timestampe to time*/
		time_t time_sec = currentTime.tv_sec;
		struct tm *format_time;
		format_time = localtime(&time_sec);
		char format_time_str[100];
		strftime(format_time_str, sizeof(format_time_str), "%Y-%m-%d %H:%M:%S", format_time);
		fprintf(timeOut,"%s:%f\n", format_time_str,(float)currentTime.tv_usec/1000);
		/* timestampe to time end*/		
	
		/* fprintf(timeOut, "%.f\n", timeInfo); */
		/* [END] Modified by smallant*/

		// Simple LOG msg.. 
		if (count % OUTPUT_GAP == 0)
		{
			gettimeofday(&endTime, 0);
			double timeuse = 1000000 * (endTime.tv_sec - startTime.tv_sec) 
					+ endTime.tv_usec - startTime.tv_usec;
			timeuse = timeuse / (1000*1000);
			printf("wrote %d bytes [msgcnt=%u](%.2f pkt/s)\n", ret, count,
				  	OUTPUT_GAP / timeuse);
			gettimeofday(&startTime, 0);

			//Get recbuf
			int rcvbuf_len;
			socklen_t getnumlen=sizeof(rcvbuf_len);
			getsockopt(sock_fd, SOL_SOCKET, SO_RCVBUF, &rcvbuf_len, &getnumlen);
			printf("the receive buf len:%d\n",rcvbuf_len);				

		}

		/* Make Sound when count == MAX_DATABLOCK_COUNT*/
		/*if (count % MAX_DATABLOCK_COUNT == 0 && count != 0)
		{
		    int set_freq = 80000 / 1000;
		    
		    int kk;

		    for(kk = 1; kk != 100; ++kk)
		    {
				ioctl(sndfd, KIOCSOUND, set_freq);
				usleep(200);

				ioctl(sndfd, KIOCSOUND, 0);
				usleep(100);
		    }
		}*/

		if(count % MAX_DATABLOCK_COUNT == 0)
		{
		    close_file();

		    /* [BEGIN] Modified by smallant 2018.1.24 */
		    if (timeOut)
		    {
			fclose(timeOut);
			timeOut = NULL;
		    }
		    /* [END] Modefied by smallant */

		    file_cnt += 1;

		    if(file_cnt > FILE_CNT)
		    {
		        printf("INFO: Collected %d files\n", FILE_CNT - start_file_cnt + 1);
		        exit(0);
		    }

		    memset(filePath, 0, sizeof(filePath));
		    sprintf(filePath, "%s%d0.dat", baseFilePath, file_cnt);
		    out = open_file(filePath, "w");

		    /* [BEGIN] Modified by smallant 2018.1.24 */
		    memset(timeFilePath, 0, sizeof(timeFilePath));
		    sprintf(timeFilePath, "%s%d0-time.txt", baseFilePath, file_cnt);
		    timeOut = open_file(timeFilePath, "w");
		    /* [END] Modefied by smallant */
			
			count = 0;
		}
		
		++count;
		
		if (ret != l)
		{
			exit_program_err(1, "fwrite");
		}
	}

	exit_program(0);
	return 0;
}

void check_usage(int argc, char** argv)
{
	if (argc != 4)
	{
		fprintf(stderr, "Usage: %s <output_file> <start_file_num> <end_file_num>\n", argv[0]);
		exit_program(1);
	}
}

FILE* open_file(char* filename, char* spec)
{
	FILE* fp = fopen(filename, spec);
	if (!fp)
	{
		perror("fopen");
		exit_program(1);
	}
	return fp;
}

//-----------------------------------------

/**
 * Close File
 *
 * @author WiX
 * @date 2015.11.17
 */ 
void close_file()
{
	if (out)
	{
		fclose(out);
		out = NULL;
	}
}

//-----------------------------------------


void caught_signal(int sig)
{
	fprintf(stderr, "Caught signal %d\n", sig);
	exit_program(0);
}

void exit_program(int code)
{
	if (out)
	{
		fclose(out);
		out = NULL;
	}
	if (sock_fd != -1)
	{
		close(sock_fd);
		sock_fd = -1;
	}
	exit(code);
}

void exit_program_err(int code, char* func)
{
	perror(func);
	exit_program(code);
}
