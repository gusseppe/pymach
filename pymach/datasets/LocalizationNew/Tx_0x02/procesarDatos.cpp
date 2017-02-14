#include<bits/stdc++.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

main(){
	string str;
	vector<string> fileFinal;
	//string nbeacons[] = {"74:DA:EA:B2:ED:76","74:DA:EA:B3:2F:4A","74:DA:EA:B4:22:18","74:DA:EA:B4:26:96","74:DA:EA:B4:3A:B3"};
	string nbeacons[] = {"C8:5B:EA:43:6F:75","FB:D3:B5:B9:89:F2","F7:24:98:9B:B6:EA","C9:EC:7A:17:D8:D5","EA:EF:87:2F:4D:93"};
	int beacons[5], cnt[5],cur=0,tot=0;
	int x,nlines=0;
	string tt;

	for(int i=0;i<5;i++) beacons[i]=cnt[i]=0;
	
	
	//lee numero de archivo , posicion
	cin>>tt;
	
	string file = "test"+tt+".csv";
	freopen(file.c_str(),"w",stdout);

	while(cin>>str>>x){
		for(int i=0;i<5;i++) 
			if( str.compare(nbeacons[i]) ==0){
				if(cnt[i]==0) cur++;
				cnt[i]++;
				beacons[i]+= x;
				if(cur==5){
					string ttStr = tt;
					cout<<tt;
					for(int j=0;j<5;j++){
						cout<<","<<(beacons[j]/cnt[j]);
						stringstream ss;
						ss << (beacons[j]/cnt[j]);
						ttStr+=","+ss.str();
					}
					cout<<endl;
					ttStr+='\n';
					fileFinal.push_back(ttStr);
					tot++;
					cur = 0;
					for(int j=0;j<5;j++) beacons[j]=0, cnt[j]=0;
					nlines++;
				}
				break;
			}
	
	}
	fclose(stdout);
	fclose(stdin);
	//Genera archivo de training
	string file2 = "train"+tt+".csv";
	string file3 = "valid"+tt+".csv";
	freopen(file2.c_str(),"w",stdout);
	
	string tmpLine;

	for(int i=0;i<(2*nlines)/3;i++){
	 	cout<<fileFinal[i];
	}
	

	fclose(stdout);

	//Genera archivo de validation
	freopen(file3.c_str(),"w",stdout);
	for(int i=(2*tot)/3;i<tot;i++){
	 	cout<<fileFinal[i];
	}

	fclose(stdout);
	//cout<<tot<<endl;
}
