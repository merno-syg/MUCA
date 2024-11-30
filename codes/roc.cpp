/**

*/

#include <iostream>
#include <sstream>
#include <fstream>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "stdio.h"

using namespace std;

// 全局变量
const int buffer = 256;

// 数据结构
struct data
{
    double score;
    int label;
};

struct dot
{
    double Sn;
    double Sp;
    double Ac;
    double Mcc;
    double CUTOFF;
};

// 函数定义
void printMessage(string s);
void printHelp(char *s);
int getRecordNumber(char* file);
void readRecord(char* file, int recordNum, struct data* myData);
void QSort(double* Array, int low, int high);
void writeResult(char* file, struct dot* myDot, int dotNumber);


// 主函数
int main(int argc ,char*argv[])
{
    char* inputFile = NULL;
    char* outputFile = NULL;

    for(int i=1; i<argc; i++)
    {
        if(strcmp(argv[i], "-i") == 0){
            inputFile = argv[i+1];
        }
        if(strcmp(argv[i], "-o") == 0){
            outputFile = argv[i+1];
        }
    }
    char recordFile[buffer] = "record.txt";
    char rocFile[buffer] = "ROCresult.txt";

    if(inputFile != NULL){
        strcpy(recordFile, inputFile);
    }
    if(outputFile != NULL){
        strcpy(rocFile, outputFile);
    }

    if(argc < 2){
        printMessage("INCORRECT PARAMETERS.");
        printHelp(argv[0]);
    }

    // 获取记录条数
    int myRecordNum =  getRecordNumber(recordFile);

    // 读入record数据
    struct data* myData = new struct data[myRecordNum];
    readRecord(recordFile, myRecordNum, myData);

    // 生成cutoff数组
    double* cutoffArray = new double[myRecordNum + 2];
    for(int i=0; i<myRecordNum; i++){
        cutoffArray[i] = myData[i].score;
    }
    cutoffArray[myRecordNum] = 1000;     // 设置最大阈值为  1000
    cutoffArray[myRecordNum+1] = -1000;  // 设置最小阈值为 -1000

    // 对阈值数组进行排序
    QSort(cutoffArray, 0, myRecordNum + 1);
    //for(int i=0; i<myRecordNum+2; i++){
    //    cout<<cutoffArray[i]<<endl;
    //}

    struct dot* myDot = new struct dot[myRecordNum + 3];
    for(int i=0; i<myRecordNum + 3; i++){
        myDot[i].Sn = myDot[i].Sp = myDot[i].Ac = myDot[i].Mcc = 0;
    }

    double lastCutoff = 10000;
    int m = 0;
    for(int i=0; i<myRecordNum + 2; i++){
        double cutOff = cutoffArray[i];
        if(cutOff == lastCutoff){
            continue;
        }
        lastCutoff = cutOff;

        double tp = 0; double fp = 0; double tn = 0; double fn = 0;
        for(int j=0; j<myRecordNum; j++){
            if(myData[j].label == 1){
                if(myData[j].score >= cutOff){ tp++; }
                else{ fn++; }
            }
            else{
                if(myData[j].score < cutOff){ tn++; }
                else{ fp++; }
            }
        }
        double sn = tp / (tp + fn);
        double sp = tn / (fp + tn);
        myDot[m].Sp = sp;
        myDot[m].Sn = sn;
        myDot[m].Ac = (tp + tn) / (tp + fn + tn + fp);
        myDot[m].Mcc = (tp * tn - fp * fn) / sqrt((tp + fp)*(tp + fn)*(tn + fp) * (tn + fn));
        myDot[m].CUTOFF = cutOff;
        m++;
    }
    //myDot[m].x = myDot[m].y = 1;

    //double myArea = calculateArea(myDot, m + 1);
    //double myArea01 = calculateArea01(myDot, m + 1);
    writeResult(rocFile, myDot, m);
    return 0;
}

// 函数体
void printMessage(string s){
    cerr<<s<<endl;
}

void printHelp(char *s)
{
    cout<<endl;
    cout<<s<<"\targuments:"<<endl;
    cout<<endl<<"    -i         record file in fasta format;  [ String ]";
    cout<<endl<<"    -o         output file [ String ]"<<endl;
    cout<<endl;
    exit(1);
}

int getRecordNumber(char* file){
    int recordNumber = 0;
    ifstream icin(file);
    if(!icin){
        printMessage("Can not open the record file!");
        exit(1);
    }
    string tmpStr;
    while(!icin.eof()){
        getline(icin, tmpStr);
        if(!tmpStr.empty()){
            if(tmpStr.at(0) == '\r' || tmpStr.at(0) == ' ' || tmpStr.at(0) == '\t'){
                continue;
            }
            recordNumber++;
        }
    }
    icin.close();
    icin.clear();
    return recordNumber;
}

void readRecord(char* file, int recordNum, struct data* myData){
    ifstream icin(file);
    if(!icin){
        printMessage("Can not open the record file!");
        exit(1);
    }
    int num = 0;
    string tmpStr;
    while(!icin.eof() && num < recordNum){
        getline(icin, tmpStr);
        if(!tmpStr.empty()){
            if(tmpStr.at(0) == '\r' || tmpStr.at(0) == ' ' || tmpStr.at(0) == '\t'){
                continue;
            }
            istringstream iStream(tmpStr);
            iStream>>myData[num].score>>myData[num].label;
            num++;
        }
    }
    icin.close();
    icin.clear();
}

void QSort(double* Array, int low, int high){
    if(low >= high){
        return;
    }
    int first  = low;
    int last = high;
    double key = Array[first];

    while(first < last){
        while(first < last && Array[last] <= key){
            last--;
        }
        Array[first] =  Array[last];
        while(first < last && Array[first] >= key){
            first++;
        }
        Array[last] = Array[first];
    }
    Array[first] = key;
    QSort(Array, low, first-1);
    QSort(Array, first+1, high);
}


void writeResult(char* file, struct dot* myDot, int dotNumber){
    ofstream ofssub(file);
    if(!ofssub){
        printMessage("Cannot open the output file.");
    }

    ofssub<<"Sn\tSp\tAcc\tMCC\t<<Cutoff"<<endl;
    for(int i=0; i<dotNumber; i++){
        ofssub<<myDot[i].Sn<<"\t"<<myDot[i].Sp<<"\t"<<myDot[i].Ac<<"\t"<<myDot[i].Mcc<<"\t"<<myDot[i].CUTOFF<<endl;
    }
    ofssub.close();
    ofssub.clear();
}
