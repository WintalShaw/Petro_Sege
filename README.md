  As I had show in this folder, we had almost 8 python files to built our system for the petroleum series segement, in this readme, we'll show how's this going on.
  
  1st:our models has two advanced models which are Gene-diffusion and Greedy_Guass,they correapond to the mydiffusion.py and gss.py.And to enhance the Greedy Guass Algorithm,we built a code in the gss_enhancer.py file which use some methods we had mentioned in our final report to improve the sensetive of GSS.
  
  To train our mydiffusion model,we need to make enough train samples to train model and checkout our system's accuracy.So we built a label make platform which is in hand_sege_label.py,if you want to make some sample be labeled,just run the hand_sege_label.py,it will shoe you a window to make label for diffusion.
  
  If you want to train our diffusion model,we give you a example file diff_train_example,it shows the basic train process of mydiffusion model,you can modify it or run it to train our reconstrcted mydiffusion.
  
  The most important, we have a full algorithm which fusion mydiffusion and enhanced Greedy Guass Segemnt to build a powerful series segement algorithm in the segemented.py.After you have trained mydiffusion model,you just need to modify the series file address,and run the segement.
  
  For user to use the algorithm convinient and intuitive,we not only integrate our algorithm,but also integrate a module to make series analysis reprot which use 1D-CNN and extraction corpora from the corpus,the detail of our platform you can get from our video.If you want tou run, just in the terminal to input streamlit run platform.py.
