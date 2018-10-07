import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Scanner;

public class KNNTextClassify {

	private String trainFile;
	private String testFile;
	
	private final String comWords="the be to of and is was its so by in that have it for not on with he as you do at this but his by from they we an or will if a b c d e f g h i j k l m n o p q r s t u v w x y z";
	Map<String, Integer> cwMap;
	
	ArrayList<Doc> trainDoc;
	ArrayList<Doc> testDoc;
	
	Map<String, Integer> words;
	Map<String, Boolean> wordsFlag;
	
	private int D;//no of documents
	private int testD;
	
	public KNNTextClassify(String trainFile, String testFile) throws FileNotFoundException {
		this.trainFile=trainFile;
		this.testFile=testFile;
		
		cwMap=new HashMap<String, Integer>();
		createCommonMap();
		
		trainDoc=new ArrayList<Doc>();
		testDoc=new ArrayList<Doc>();
		
		words=new HashMap<String, Integer>();
		wordsFlag=new HashMap<String, Boolean>();
		
		parseFile(this.trainFile,trainDoc);
		System.out.print("Train size: ");
		System.out.println(trainDoc.size());
		
		parseFile(this.testFile,testDoc);
		System.out.print("Test size: ");
		System.out.println(testDoc.size());
		
		testD=testDoc.size();
		D=trainDoc.size()+testDoc.size();
		
		System.out.print("words: ");
		System.out.println(words.size());
		
		System.out.println("Reading complete!");
		
		createTfIdf(trainDoc);
		createTfIdf(testDoc);
		System.out.println("TF-IDF complete!");
		
		kNN(trainDoc, testDoc);
		
		//System.out.println("the: "+words.get("the"));
		//System.out.println("is: "+words.get("is"));
		//System.out.println(words);
		
		
//		int size=testDoc.size();
//		for(int i=0;i<size;i++){
//			System.out.println(i+": "+testDoc.get(i).label);
//		}
	}
	
	private void createCommonMap(){
		String[] wordStr=comWords.split(" ");
		//System.out.println(wordStr.length);
		int len=wordStr.length;
		for(int i=0;i<len;i++){
			cwMap.put(wordStr[i], 1);
		}
		cwMap.put(" ",1);
		cwMap.put("",1);
		//System.out.println(cwMap.size());
	}
	
	private void parseFile(String fileName, ArrayList<Doc> docList) throws FileNotFoundException{
		Scanner input=new Scanner(new File(fileName));
		String line="";
		while(input.hasNext()){
			String str="";
			Doc doc=new Doc();
			line=input.nextLine();//label
			doc.label=new String(line);
			line=input.nextLine();//blank
			//read title
			while(true){
				line=input.nextLine();
				if(line.equals("")){
					doc.title=str;
					break;
				}
				str+=(line+" ");
			}
			
			//read address
			while(true){
				line=input.nextLine();
				if(line.equals("")){
					break;
				}
			}
			
			//read data	
			while(true){
				if(input.hasNext()){
					line=input.nextLine();
				}
				else{
					break;
				}
				if(line.equals("")){
					if(input.hasNext())
						line=input.nextLine();
					break;
				}
				str+=(line+" ");
			}
			
			doc.eucVec=new HashMap<String, Integer>();
			doc.tfIdf=new HashMap<String, Double>();
			
			str=str.toLowerCase();
			String[] strs=str.split("\\W+|\\d+");
			int length=strs.length;
			for(int i=0;i<length;i++){
				if(strs[i].equals(""))
				//if(cwMap.containsKey(strs[i]))
					continue;
				if(words.containsKey(strs[i]) && wordsFlag.get(strs[i])==true){
					int val=words.get(strs[i]);
					words.put(strs[i], val+1);
					wordsFlag.put(strs[i],false);
				}
				else if(!(words.containsKey(strs[i]))){
					words.put(strs[i], 1);
					wordsFlag.put(strs[i],false);
				}
				
				if(doc.eucVec.containsKey(strs[i])){
					int val=doc.eucVec.get(strs[i]);
					doc.eucVec.put(strs[i], val+1);
				}
				else{
					doc.eucVec.put(strs[i], 1);
				}
				
			}
			
			for(String key:wordsFlag.keySet()){
				wordsFlag.put(key, true);
			}
			
			
			//System.out.println(doc.hamVec);
			//System.out.println(doc.eucVec);
			
			docList.add(doc);
		}
		
		input.close();
	}
	
	private void createTfIdf(ArrayList<Doc> docList){
		int size=docList.size();
		for(int i=0;i<size;i++){
			double tf,cw,idf,tfIdfVal;
			int wd=docList.get(i).eucVec.size();
			for(String key:docList.get(i).eucVec.keySet()){
				tf=docList.get(i).eucVec.get(key)/(wd*1.0);
				cw=words.get(key);
				idf=Math.log((D*1.0)/cw);
				tfIdfVal=tf*idf;
				docList.get(i).tfIdf.put(key, tfIdfVal);
			}
		}
	}
	
	private void kNN(ArrayList<Doc> train, ArrayList<Doc> test){
		PriorityQueue<Classifier> hamPQ=new PriorityQueue<Classifier>(20, new Comparator<Classifier>(){
			public int compare(Classifier c1, Classifier c2){
				return c1.value-c2.value;
			}
		});
		
		PriorityQueue<Classifier> eucPQ=new PriorityQueue<Classifier>(20, new Comparator<Classifier>(){
			public int compare(Classifier c1, Classifier c2){
				return c1.value-c2.value;
			}
		});
		
		PriorityQueue<Classifier> tfIdfPQ=new PriorityQueue<Classifier>(20, new Comparator<Classifier>(){
			public int compare(Classifier c1, Classifier c2){
				if((c2.ti-c1.ti)>0)
					return 1;
				if((c2.ti-c1.ti)<0)
					return -1;
				else
					return 0;
			}
		});
		
		int[] accHamK=new int[3];
		int[] accEucK=new int[3];
		int[] accTfIdfK=new int[3];
		
		int trainSize=train.size();
		int testSize=test.size();
		double[] tempVal;
		
		for(int i=0;i<testSize;i++){
			if(i%100==0)
				System.out.println(i);
			
			hamPQ.clear();
			eucPQ.clear();
			tfIdfPQ.clear();
			
			double testD=0;
			for(String key:test.get(i).tfIdf.keySet()){
				testD+=Math.pow(test.get(i).tfIdf.get(key),2);
			}
			
			for(int j=0;j<trainSize;j++){
				tempVal=vectorCalculation(train.get(j),test.get(i),testD);
				//System.out.println(tempVal);
				//Classifier cl=new Classifier(train.get(j).label,tempVal);
				//System.out.println(cl);
				//PQ.add(cl);
				//System.out.println(PQ.size());
				//System.out.println(PQ);
				hamPQ.add(new Classifier(train.get(j).label,(int)tempVal[0],0));
				eucPQ.add(new Classifier(train.get(j).label,(int)tempVal[1],0));
				tfIdfPQ.add(new Classifier(train.get(j).label,0,tempVal[2]));
			}
			
			//hamming
			String[] labels=getLable(hamPQ);
//			for(String st:labels){
//				System.out.println(st);
//			}
//			System.out.println("");
			String testLab=test.get(i).label;
			for(int k=0;k<3;k++){
			if(testLab.equals(labels[k]))
				accHamK[k]++;
			}
			
			//euclidean
			labels=getLable(eucPQ);
			testLab=test.get(i).label;
			for(int k=0;k<3;k++){
			if(testLab.equals(labels[k]))
				accEucK[k]++;
			}
			
			//tfidf
			labels=getLable(tfIdfPQ);
			testLab=test.get(i).label;
			for(int k=0;k<3;k++){
				if(testLab.equals(labels[k]))
					accTfIdfK[k]++;
			}
		}
		
		System.out.println("\t\tk=1\tk=3\tk=5");
		
		System.out.print("Hamming: \t");
		//System.out.println("k=1\tk=3\tk=5");
		for(int k=0;k<3;k++){
			//System.out.print(accHamK[k]/(testD*1.0)+"%\t");
			System.out.printf("%.2f%%\t",100*accHamK[k]/(testD*1.0));
		}
		System.out.println("");
		
		System.out.print("Euclidean: \t");
		//System.out.println("k=1\tk=3\tk=5");
		for(int k=0;k<3;k++){
			//System.out.print(accEucK[k]/(testD*1.0)+"%\t");
			System.out.printf("%.2f%%\t",100*accEucK[k]/(testD*1.0));
		}
		System.out.println("");
		
		System.out.print("Cosine: \t");
		//System.out.println("k=1\tk=3\tk=5");
		for(int k=0;k<3;k++){
			//System.out.print(accTfIdfK[k]/(testD*1.0)+"%\t");
			System.out.printf("%.2f%%\t",100*accTfIdfK[k]/(testD*1.0));
		}
		System.out.println("");
		
		//System.out.println(tfIdfPQ);
	}
	
	private double[] vectorCalculation(Doc trainDoc, Doc testDoc, double testD){
		double[] distance=new double[3];
		int trainSize=trainDoc.eucVec.size();
		int testSize=testDoc.eucVec.size();
		
		Doc train=new Doc(trainDoc);
		Doc test=new Doc(testDoc);
		
		double trainD=0;
		//double testD=0;
		
		int hamDis=0;
		int eucDis=0;
		double similarity=0;
		for(String key:test.eucVec.keySet()){
			//testD+=Math.pow(test.tfIdf.get(key),2);
			if(train.eucVec.containsKey(key)){
				hamDis++;
				eucDis+=(Math.pow((train.eucVec.get(key)-test.eucVec.get(key)),2));
				similarity+=train.tfIdf.get(key)*test.tfIdf.get(key);
				train.eucVec.remove(key);
				//test.eucVec.remove(key);
				
				trainD+=Math.pow(train.tfIdf.get(key),1);
			}
			else{
				eucDis+=Math.pow(test.eucVec.get(key), 2);
			}
		}
		for(String key:train.eucVec.keySet()){
			eucDis+=Math.pow(train.eucVec.get(key), 2);
			trainD+=Math.pow(train.tfIdf.get(key),2);
		}
		//System.out.println(count);
		//System.out.println(testSize+" "+count+" "+trainSize+" "+count);
		//return (testSize-hamDis+trainSize-hamDis);
		double trainTestVec=Math.sqrt(trainD)*Math.sqrt(testD);
		similarity=similarity/trainTestVec;
		
		distance[0]=testSize-hamDis+trainSize-hamDis;
		distance[1]=eucDis;
		distance[2]=similarity;
		return distance;
	}
	
	private String[] getLable(PriorityQueue<Classifier> PQ){
		String[] labels=new String[3];
		Map<String, Integer> labelMap=new HashMap<String, Integer>();
		Classifier cl=PQ.peek();
		labels[0]=cl.label;
		//System.out.println(PQ.size());
		for(int i=0;i<5;i++){
			cl=PQ.poll();
			String str=new String(cl.label);
			if(labelMap.containsKey(str)){
				int val=labelMap.get(str);
				labelMap.put(str,val+1);
			}
			else{
				labelMap.put(str,1);
			}
			if(i==2){
				String lab=mostOccurredLabel(labelMap);
				labels[1]=lab;
				//System.out.println(lab);
			}
		}
		
		String lab=mostOccurredLabel(labelMap);
		labels[2]=lab;
		//System.out.println(lab);
		
		
//		System.out.println(labelMap);
//		labelMap=sortByValue(labelMap);
//		System.out.println(labelMap);
		
		
		return labels;
	}
	
	private String mostOccurredLabel(Map<String, Integer> labelMap){
		int best=-1;
		String bestLabel="";
		for(String key:labelMap.keySet()){
			if(labelMap.get(key)>best){
				best=labelMap.get(key);
				bestLabel=key;
			}
		}
		return bestLabel;
	}
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO Auto-generated method stub
		String trainFile="//Users//ahmadsabbir//Documents//workspace//KNNTextClassification//resource//training.data";
		String testFile="//Users//ahmadsabbir//Documents//workspace//KNNTextClassification//resource//test.data";
		new KNNTextClassify(trainFile, testFile);
	}

}

class Doc{
	String label;
	String title;
	String address;
	HashMap<String, Integer> eucVec;
	HashMap<String, Double> tfIdf;
	
	public Doc(){
		this.label="";
		this.title="";
		this.address="";
	}
	
	public Doc(Doc doc){
		this.label=new String(doc.label);
		this.title=new String(doc.title);
		this.address=new String(doc.address);
		this.eucVec=new HashMap<String, Integer>(doc.eucVec);
		this.tfIdf=new HashMap<String, Double>(doc.tfIdf);
	}
}

class Classifier{
	String label;
	int value;
	double ti;
	public Classifier(String label, int value,double ti){
		this.label=label;
		this.value=value;
		this.ti=ti;
	}
	
	public String toString(){
		String str="<"+label+","+value+","+ti+">";
		return str;
	}
}