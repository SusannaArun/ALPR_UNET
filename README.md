•	 Title of the Project: 

o	Automatic License Plate Recognition (ALPR) using U-Net

•	Goal(s) of the project:

o	 Create a model that automatically reads license plates by utilizing U-Net for segmentation of letters and numbers from images

o	Evaluate the efficiency and practicality of using U-Net in segmentation

•	Background of the project:

o	Index Terms: Computer Vision, Artificial Intelligence, Natural Language Processing (NLP), Medical Research, Robotics, Classification, Segmentation, Optical Character Recognition (OCR)

o	This project combines two classical computer vision problems (Image Segmentation and Optical Character Recognition) and applies them to a real world problem, automatic license plate recognition (ALPR). ALPR is currently deployed in a variety of circumstances, including automatic toll collection, parking enforcement, and road law enforcement in the form of speed cameras. The problem can be broken into two subproblems. Firstly, the system must differentiate the license plate from other image artifacts. Secondly, the system must correctly decode the characters represented on the license plate, in the correct order. The first problem is a classic example of image segmentation, where a system will output coordinates of a box containing the license plate. The second is one of the oldest computer vision classification problems. Each character must be correctly placed into classes representing each character of the alphabet the system is trying to analyze, as well as any numerals which may be contained in the class set. 

o	
•	Reference papers:

	Laroca, R., Zanlorensi, L.A., Gonçalves, G.R., Todt, E., Schwartz, W.R., & Menotti, D. (2019). An Efficient and Layout-Independent Automatic License Plate Recognition System Based on the YOLO detector. ArXiv, abs/1909.01754.
	Chowdhury, P.N., Shivakumara, P., Raghavendra, R., Pal, U., Lu, T., & Blumenstein, M. (2019). A New U-Net Based License Plate Enhancement Model in Night and Day Images. ACPR.

	Panahi, R., & Gholampour, I. (2016). Accurate detection and recognition of dirty vehicle plate numbers for high-speed applications. IEEE Transactions on intelligent transportation systems, 18(4), 767-779.

	E. Irwansyah, Y. Heryadi and A. A. Santoso Gunawan, "Semantic Image Segmentation for Building Detection in Urban Area with Aerial Photograph Image using U-Net Models," 2020 IEEE Asia-Pacific Conference on Geoscience, Electronics and Remote Sensing Technology (AGERS), 2020, pp. 48-51, doi: 10.1109/AGERS51788.2020.9452773.
	Kumar, G V S & .R, Vijaya Kumar. (2014). Review on Image Segmentation Techniques. International Journal of Scientific Research Engineering & Technology (IJSRET). 3. 992-997. 

	S. Minaee, Y. Y. Boykov, F. Porikli, A. J. Plaza, N. Kehtarnavaz and D. Terzopoulos, "Image Segmentation Using Deep Learning: A Survey," in IEEE Transactions on Pattern Analysis and Machine Intelligence, doi: 10.1109/TPAMI.2021.3059968. 

•	Deep learning model to be used as a base model:

o	Tung, Chun-Liang & Wang, Ching-Hsin & Peng, Bo-Syuan. (2021). A Deep Learning Model of Dual-Stage License Plate Recognition Applicable to the Data Processing Industry. Mathematical Problems in Engineering. 2021. 10.1155/2021/3723715.

o	K, B., 2022. U-Net Architecture for Image Segmentation. [online] Paperspace Blog. Available at: <https://blog.paperspace.com/unet-architecture-image-segmentation/> [Accessed 29 March 2022].

•	Experiments you are planning to run:

o	Letter and number classification accuracy for OCR by evaluating on License Plate dataset

o	MAP (mean average precision) for image segmentation

•	Dataset(s) to be used in the project. Give a brief description about the dataset:

o	For this project, the Rodosol-Alpr dataset will be used. This dataset consists of 20,000 images that were captured at toll booths in Brazil. The dataset consists of license plates captured on different types of vehicles during day and nighttime as well as in varied weather conditions. All images are consistently 1280 x 720 pixels. The dataset also consists of two different license plate layouts: Brazil and Mercosur. Brazilian license plates contain 3 letters followed by four digits whereas Mercosur license plates contain 3 letters,1 digit, followed by 1 letter and 2 digits.

•	Dataset References: 

o	Notebook: https://www.kaggle.com/code/firebee/anpr-using-unet/notebook
o	Dataset: https://github.com/raysonlaroca/rodosol-alpr-dataset/
o	Sample project: https://github.com/JungUnYun/License-Plate-Recognition
