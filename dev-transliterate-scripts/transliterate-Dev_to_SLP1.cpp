#include <string>
#include <cstring>
#include <regex>
#include <sstream>
#include <fstream>
#include <iostream>
using namespace std;

string ReplaceString(string subject, const string& search, const string& replace)
{
	size_t pos=0;
	while((pos=subject.find(search, pos)) != string::npos)
	{
		subject.replace(pos, search.length(),replace);
		pos+=1;
	}
	return subject;
}

string ReplaceStringRestricted(string subject, const string& search, const string& replace)
{
	size_t pos=0,p=0;
	char c;
	string replace_new=replace+"a";
	subject+=" ";
	// replace_new=replace_new+m;
	while((pos=subject.find(search, pos)) != string::npos)
	{
		c=subject.at(pos+3);
		if(c=='A' || c=='i' || c=='I' || c=='u' || c=='U' || c=='f' || c=='F' || c=='x' || c=='X' || c=='e' || c=='E' || c=='o' || c=='O')
			subject.replace(pos, search.length(),replace);
		else
			subject.replace(pos, search.length(),replace_new);
		pos+=1;
	}
	subject.erase(subject.size()-1);
	return subject;
}

int main(int argc, char const *argv[])
{
	string s;
	string filename="1";
	string vowel_dn[]={"अ","आ","इ","ई","उ","ऊ","ऋ","ॠ","ऌ","ॡ","ए","ऐ","ओ","औ","ं","ः","ँ","ᳲ","ᳳ"};
	string vowel_dn_joiner[]={"ा","ि","ी","ु","ू","ृ","ॄ","ॢ","ॣ","े","ै","ो","ौ"};
	string consonants_dn[]={"क","ख","ग","घ","ङ","च","छ","ज","झ","ञ","ट","ठ","ड","ढ","ण","त","थ","द","ध","न","प","फ","ब","भ","म","य","र","ल","व","श","ष","स","ह","ळ"};
	string consonants_dn_halanta[]={"क्","ख्","ग्","घ्","ङ्","च्","छ्","ज्","झ्","ञ्","ट्","ठ्","ड्","ढ्","ण्","त्","थ्","द्","ध्","न्","प्","फ्","ब्","भ्","म्","य्","र्","ल्","व्","श्","ष्","स्","ह्","ळ्"};
	string vowel_slp1[]={"a","A","i","I","u","U","f","F","x","X","e","E","o","O","M","H","~","Z","V"};
	string consonants_slp1[]={"k","K","g","G","N","c","C","j","J","Y","w","W","q","Q","R","t","T","d","D","n","p","P","b","B","m","y","r","l","v","S","z","s","h","L"};
	if(argc==2)
	{
		filename=argv[1];
	}
	else
	{
		cout<<"Please enter the filename, which has data in Devanagari : \n";
		cin>>filename;
	}
	ifstream file;
	file.open(filename);
	while(getline(file,s))
	{
		for(int i=0;i<34;i++)
		{
			s=ReplaceString(s,consonants_dn_halanta[i],consonants_slp1[i]);
		}
		for(int i=0;i<13;i++)
		{
			s=ReplaceString(s,vowel_dn_joiner[i],vowel_slp1[i+1]);
		}
		for(int i=0;i<34;i++)
		{
			s=ReplaceStringRestricted(s,consonants_dn[i],consonants_slp1[i]);
		}
		for(int i=0;i<19;i++)
		{
			s=ReplaceString(s,vowel_dn[i],vowel_slp1[i]);
		}
		cout<<s<<endl;
	}

	
	return 0;
}
