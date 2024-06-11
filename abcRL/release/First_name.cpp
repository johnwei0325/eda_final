#include<iostream>
#include<fstream>
#include<string>
#include<vector>
using namespace std;
void print(vector<string> a){
	for(int i=0;i<a.size();i++){
		cout<<a[i]<<endl;
	} cout<<endl;
}
void print(vector<int> a){
	for(int i=0;i<a.size();i++){
		cout<<a[i]<<" ";
	} cout<<endl;
}
struct node{
	string f, name, left, right;
};
node NewNode(string F, string Name, string Left){
	node newnode; newnode.f = F; newnode.name = Name; newnode.left = Left; newnode.right = "";
	return newnode;
}
node NewNode(string F, string Name, string Left, string Right){
	node newnode; newnode.f = F; newnode.name = Name; newnode.left = Left; newnode.right = Right;
	return newnode;
}
int main(){
	vector<string> text;
	vector<string> no_space;
	int init;
	string empty = "";
	ifstream fin("release/release/netlists/design1.v");
	ofstream fout("release/release/netlists/design1_m.v");
	vector<int> index;
	int size = 100; char num[size]; vector<int> pos;
	while(fin.getline(num,size)){
		text.push_back(num);
	}
	for(int i=0;i<text.size();i++){
		no_space.push_back(empty);
		for(int j=0;j<text[i].size();j++){
			if(text[i][j]!=' '){
				no_space[i].push_back(text[i][j]);
			}
		}
	}
	for(int i=0;i<no_space.size();i++){
		if(no_space[i].substr(0,2)=="or"||no_space[i].substr(0,2)=="na"||no_space[i].substr(0,2)=="bu"||no_space[i].substr(0,2)=="no"||no_space[i].substr(0,2)=="an"||no_space[i].substr(0,2)=="xo"||no_space[i].substr(0,2)=="nx"){
			init=i;
			break;
		}		
	}
	fout<<"// Benchmark"<<endl;
	fout<<"module"<<endl;
	vector<int> parse, empty_int;
	for(int i=0;i<init;i++){
		if(no_space[i].substr(0,6)=="module"){}
		else if(no_space[i].substr(0,2)=="//"){}
		else if(no_space[i].substr(0,5)=="input"){
			fout<<no_space[i].substr(0,5)<<" "<<no_space[i].substr(5,no_space[i].size())<<endl;
		}
		else if(no_space[i].substr(0,6)=="output"){
			fout<<no_space[i].substr(0,6)<<" "<<no_space[i].substr(6,no_space[i].size())<<endl;
		}
		else if(no_space[i].substr(0,4)=="wire"){
			fout<<no_space[i].substr(0,4)<<" "<<no_space[i].substr(4,no_space[i].size())<<endl;
		}		
		else{
			fout<<no_space[i]<<endl;
		}
	}
	for(int i=init;i<no_space.size()-1;i++){
		parse = empty_int;
		for(int j=0;j<no_space[i].size();j++){
			if(no_space[i][j]=='('||no_space[i][j]==','||no_space[i][j]==')'){
				parse.push_back(j);
			}
		}
		fout<<"assign "<<no_space[i].substr(parse[0]+1,parse[1]-parse[0]-1)<<" = ";
		if(no_space[i].substr(0,2)=="or"){
			fout<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" | ";
			fout<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,2)=="an"){
			fout<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" & ";
			fout<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,2)=="xo"){
			fout<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" ^ ";
			fout<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,2)=="xn"){
			fout<<"~"<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" ^ ";
			fout<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,3)=="nor"){
			fout<<"~"<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" & ";
			fout<<"~"<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,3)=="nan"){
			fout<<"~"<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<" | ";
			fout<<"~"<<no_space[i].substr(parse[2]+1,parse[3]-parse[2]-1)<<";";
		}
		else if(no_space[i].substr(0,3)=="not"){
			fout<<"~"<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<";";
		}
		else if(no_space[i].substr(0,3)=="buf"){
			fout<<no_space[i].substr(parse[1]+1,parse[2]-parse[1]-1)<<";";
		}		
		fout<<endl;
	}
	fout<<no_space[no_space.size()-1];
}
