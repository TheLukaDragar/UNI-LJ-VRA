import os
import argparse
import numpy as np




class Result():
    def __init__(self, test1,test2,test3,name,fn1,fn2,fn3):
        self.test1 = test1
        self.test2 = test2
        self.test3 = test3
        self.name = name
        self.fn1 = fn1
        self.fn2 = fn2
        self.fn3 = fn3
        self.summary=None
        self.weight=None
        self.mae=None

    def set_summary(self,summary):
        self.summary=summary

    def set_weight(self,weight):
        self.weight=weight

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Combine predictions of different models")
    parser.add_argument("--eva", type=str, default="./predictions/37orwro0/7327/", help="eva prediction folder")
    parser.add_argument("--convnext", type=str, default="./predictions/y23waiez/32585/", help="convnext prediction folder")
    parser.add_argument("--weight", type=float, default=0.75, help="weight of convnext prediction")
    parser.add_argument("--output", type=str, default="./predictions/combined/", help="output folder")
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    # load predictions
    eva_prediction = None
    convnext_prediction = None


    names=["Test1_preds.txt","Test2_preds.txt","Test3_preds.txt"]

    test1, test2, test3 = [], [], []
    fn1, fn2, fn3 = [], [], []

    with open(os.path.join(args.eva,names[0])) as f1:
        for line in f1:
            l=line.split(",")
            fn1.append(l[0])
            test1.append(float(l[1]))

    with open(os.path.join(args.eva,names[1])) as f2:
        for line in f2:
            l=line.split(",")
            fn2.append(l[0])
            test2.append(float(l[1]))

    with open(os.path.join(args.eva,names[2])) as f3:
        for line in f3:
            l=line.split(",")
            fn3.append(l[0])
            test3.append(float(l[1]))

    eva_prediction = Result(test1,test2,test3,"eva",fn1,fn2,fn3)


    test1, test2, test3 = [], [], []
    fn1, fn2, fn3 = [], [], []

    with open(os.path.join(args.convnext,names[0])) as f1:
        for line in f1:
            l=line.split(",")
            fn1.append(l[0])
            test1.append(float(l[1]))

    with open(os.path.join(args.convnext,names[1])) as f2:
        for line in f2:
            l=line.split(",")
            fn2.append(l[0])
            test2.append(float(l[1]))

    with open(os.path.join(args.convnext,names[2])) as f3:
        for line in f3:
            l=line.split(",")
            fn3.append(l[0])
            test3.append(float(l[1]))

    convnext_prediction = Result(test1,test2,test3,"convnext",fn1,fn2,fn3)

    print(f"loaded Eva prediction with {len(eva_prediction.test1)+len(eva_prediction.test2)+len(eva_prediction.test3)} samples")
    print(f"loaded ConvNext prediction with {len(convnext_prediction.test1)+len(convnext_prediction.test2)+len(convnext_prediction.test3)} samples")

    # combine predictions
    assert len(eva_prediction.test1)==len(convnext_prediction.test1)
    assert len(eva_prediction.test2)==len(convnext_prediction.test2)
    assert len(eva_prediction.test3)==len(convnext_prediction.test3)

    #turn into numpy array
    eva_prediction.test1=np.array(eva_prediction.test1)
    eva_prediction.test2=np.array(eva_prediction.test2)
    eva_prediction.test3=np.array(eva_prediction.test3)

    convnext_prediction.test1=np.array(convnext_prediction.test1)
    convnext_prediction.test2=np.array(convnext_prediction.test2)
    convnext_prediction.test3=np.array(convnext_prediction.test3)

    #combine using weight
    combined_test1=eva_prediction.test1*(1-args.weight)+convnext_prediction.test1*args.weight
    combined_test2=eva_prediction.test2*(1-args.weight)+convnext_prediction.test2*args.weight
    combined_test3=eva_prediction.test3*(1-args.weight)+convnext_prediction.test3*args.weight

    #save combined predictions
    with open(os.path.join(args.output,"Test1_preds.txt"),"w") as f1:
        for i in range(len(eva_prediction.test1)):
            f1.write(f"{eva_prediction.fn1[i]},{combined_test1[i]}\n")

    with open(os.path.join(args.output,"Test2_preds.txt"),"w") as f2:
        for i in range(len(eva_prediction.test2)):
            f2.write(f"{eva_prediction.fn2[i]},{combined_test2[i]}\n")

    with open(os.path.join(args.output,"Test3_preds.txt"),"w") as f3:
        for i in range(len(eva_prediction.test3)):
            f3.write(f"{eva_prediction.fn3[i]},{combined_test3[i]}\n")

    print(f"saved combined predictions to {args.output} with weight {args.weight} for convnext and {1-args.weight} for eva")

    

    
