import numpy as np
import cv2
from numpy import linalg as la
import statistics
from matplotlib import pyplot as plt
import random


print("ACEST PROGRAM VA CAUTA AUTOMAT")
print("PERSOANA ALEASA DUPA POZA 10")
print("IN FUNCTIE DE ALEGERILE")
print("PE CARE UTILIZATORUL LE VA FACE\n")

alg=input("Alege algoritmul dorit dintre:'nn' sau 'knn'\n")

if ((alg!="nn") and (alg!="knn")):
    print("Eroare nu ai introdus datele corect !!!")
    
else:
       
    #x este nr poze de antrenare
    x=int(input("Alege numarul de poze de antrenare, de preferat:6, 8, 9\n"))

    A=np.zeros([10304,40*x])
    #AT&T Database of Faces
    caleBD=r'F:\zSCOALA\acs folder\proiect\att_faces' #calea catre poza

    for i in range(1,41):
        caleFolderPers = caleBD + '\s'+str(i) +'\\'
        for j in range(1,x+1):
            calePozaAntrenare=caleFolderPers+str(j)+'.pgm'
            pozaAntrenare=np.array(cv2.imread(calePozaAntrenare,0))
            pozaVect=pozaAntrenare.reshape(10304,)
            A[:, x*(i-1)+(j-1)] = pozaVect
    
    vr=input("Doresti sa alegi numarul persoanei cautate ?\nAlege 'da' sau 'nu'.\n")
    
    if vr=='da':
        nr=int(input("Alege persoana cautata de la 1 la 40.\n"))
        #pentru alegere manuala  
        cFP = caleBD + '\s'+str(nr) +'\\'    
        calePozaCautata = cFP+str(10)+'.pgm'
    
    else:
        print("Automat se va alege persoana 8.\n")
        #pentru alegere automata
        calePozaCautata=r'F:\zSCOALA\acs folder\proiect\att_faces\s8\10.pgm'  #calea catre poza
    
    pozaCautata=np.array(cv2.imread(calePozaCautata,0))
    pozaCautataVect=pozaCautata.reshape(-1,)
    
    
    show=input("Doresti sa vezi imaginea persoanei alese ?\nAlege 'da' sau 'nu'.\n")
    
    if show=="da":
        plt.imshow(pozaCautataVect.reshape(112,92), cmap='gray',vmin=0, vmax=255) 
        plt.show() 



        
    def AlgortimNN(A,pozaCautataVect,norm):
        
        z=np.zeros(len(A[0]))
        for i in range (0,len(A[0])):
        
            if norm=="2":
                z[i]=la.norm(A[:,i]-pozaCautataVect)
            
            elif norm=="1":
                z[i]=la.norm(A[:,i]-pozaCautataVect,1)
            
            elif norm=="infinit":
                z[i]=la.norm(A[:,i]-pozaCautataVect,np.inf)
            
            elif norm=="cos":
                z[i]=1-np.dot(A[:,i],pozaCautataVect)/(la.norm(A[:,i])*la.norm(pozaCautataVect))
        
            
            
            pozitiaCautata=np.argmin(z)
        
        return pozitiaCautata  


    def AlgortimKNN(A,pozaCautataVect,k,norm):
        z=np.zeros(len(A[0]))
        for i in range (0,len(A[0])):
            if norm=="2":
                z[i]=la.norm(A[:,i]-pozaCautataVect)
            
            elif norm=="1":
                z[i]=la.norm(A[:,i]-pozaCautataVect,1)
            
            elif norm=="infinit":
                z[i]=la.norm(A[:,i]-pozaCautataVect,np.inf)
            
            elif norm=="cos":
                z[i]=1-np.dot(A[:,i],pozaCautataVect)/(la.norm(A[:,i])*la.norm(pozaCautataVect))
                
            pozitii=np.argsort(z)
    
        idk=pozitii[:k]

        return idk 
        
    

    if alg=="nn":
        
        norm=input("Introdu norma dorita dintre:1 ,2 ,infinit ,cos\n")
        if ((norm!="1") and (norm!="2")and (norm!="cos")and (norm!="infinit")):
            print("NU ai introdus o norma buna !!!")
            
        else :
            poz=AlgortimNN(A, pozaCautataVect, norm)
            print()
            print("Nr poza",poz)

            #afisarea pozei gasite NN
            plt.imshow(A[:,poz].reshape(112,92), cmap='gray',vmin=0, vmax=255)
            plt.show()  



    elif alg=="knn":
        
        k=int(input("Introdu valoare pentru 'K', de preferat:\n1(nn), 3, 5, 7, 9.\n"))
        
        
        norm=input("Introdu norma dorita dintre:1 ,2 ,infinit ,cos\n")
        
        
        if ((norm!="1") and (norm!="2")and (norm!="cos")and (norm!="infinit")):
            print("NU ai introdus o norma buna !!!")
            
        else :
            pozz=AlgortimKNN(A, pozaCautataVect,k,norm)
        
        
        
            #alegere=input("Doresti afisarea pozelor K gasite de algoritmul KNN ?\nAlege:'da' sau 'nu'.\n")
        
            #afisarea tuturor pozelor gasite K-NN
            #if alegere=="da":
               # print(pozz)
                
                #for i in range (0,len(pozz)):   
                        #plt.imshow(A[:,pozz[i]].reshape(112,92), cmap='gray',vmin=0, vmax=255)
                        #plt.show() 


            pozzPers=list(pozz)

            #schimb nr pozei in numarul persoanei (se inparte la 8(x) nu la 10 pentru ca depinde de matrice)
            for i in range (0,len(pozz)):
                pozz[i]=pozz[i]//x+1

            #pt a vedea sirul de persoane gasite
            #print(pozz)

            #verific ce persoana apare cel mai des
            nrPersCautata=statistics.mode(pozz) 
            #print(nrPersCautata)
        
            print()
            print("Persoana gasita are numarul:",nrPersCautata)
            print()
        
        
        #preluarea unei poze a persoanei cautate, dintre pozele gasite (prima cu break//ultima fara break)
        def gasire():
    
            for i in range (0,len(pozz)):
                if pozz[i]==nrPersCautata:
                       pozaGasita=i
                       break
        
    
            #afisarea pozei selectate K-NN dintre cele gasite
            plt.imshow(A[:,pozzPers[pozaGasita]].reshape(112,92), cmap='gray',vmin=0, vmax=255)
            plt.show() 

        # alegerea pozei random
        def gasireRandom():
            rand=random.randint(x*(nrPersCautata-1),(x*nrPersCautata)-1)
    
            # alegerea pozei random din intervalul cu indicile (persoanei-1) trebuie inmultin cu x fiind primul capat din stanga
            # pentru capatul din dreapta indicile persoanei* x - 1
            # pt ca vorbim de matricea A
            # ex pt x=8:p1:0-7 p2:8-15 p3:16-23
            # x*(nrPers-1),(x*nrPers)-1
            # x=nr poze de antrenare
        
            #afisarea pozei selectate K-NN dintre cele existente aleasa random
            plt.imshow(A[:,rand].reshape(112,92), cmap='gray',vmin=0, vmax=255)
            plt.show() 
              
        y=input("Alege din ce grup doresti afisarea finala, a persoanei gasite:\n1-random \n2-dintre pozele asociate de program cu persoana cautata\n3-nu doresti afisare.\n")
              
        if y=="2":
            gasire()
        elif y=="1":
            gasireRandom()
        elif y=="3":
            print("Ok")
        else: print("Nu ai ales corect !!!")



