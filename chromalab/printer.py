import os
import numpy as np
from PIL import Image, ImageDraw, ImageCms, ImageFont
import packcircles as pc
import pickle
# os.environ["IMAGEIO_FFMPEG_EXE"] = "/opt/homebrew/Cellar/ffmpeg/6.0_1/"
font = ImageFont.load_default()
class PrinterMapper:
    def __init__(self,channelPrinterMap,size,printNames=None, directory = "."):
        self.printCount = len(channelPrinterMap((0,0,0,0,0,0)))
        self.channelPrinterMap = channelPrinterMap
        self.printerImages = [Image.new('CMYK',size) for i in range(self.printCount)]
        self.printerDraws = [ImageDraw.Draw(image) for image in self.printerImages]
        self.printNames = printNames
        self.directory = directory

        m = channelPrinterMap((1,2,3,4,5,6))
        channel2Printer = {}
        for channel in range(1,7):
            for i, cmyk in enumerate(m):
                if channel in cmyk:
                    channel2Printer[channel] = i
        self.channel2Printer = channel2Printer

    def MapDraw(self, color):
        colors = self.channelPrinterMap(color)
        return self.printerDraws, colors
    
    def DrawLabel(self, color, text_placement_x, text_placement_y):
        if np.isnan(color[0]):
            self.printerDraws[0].text((text_placement_x, text_placement_y), "NaN", font=font, fill=(0,0,0,255))
            return
        for i, value in enumerate(color):
            draw = self.printerDraws[self.channel2Printer[i+1]]
            draw.text((text_placement_x + ((i%3)*20), text_placement_y + ((i//3)*10)), f"{str(int(value*100))}," , font=font, fill=(0,0,0,255))

    def DrawText(self, text, text_placement_x, text_placement_y):
        self.printerDraws[0].text((text_placement_x,text_placement_y),text,font=font,fill=(0,0,0,255))

    def DrawEllipse(self, circle, color):
        if np.isnan(color[0]):
            print("printing NAN")
            self.printerDraws[0].ellipse(circle, width=3, fill=(0,0,0,25),outline=(0,0,0,255))
            return
        color = [int(c*255) for c in color]
        colors = self.channelPrinterMap(color)
        # print(f'colors= {colors}')
        for i, draw in enumerate(self.printerDraws):
            draw.ellipse(circle,width=0,fill=colors[i])

    def DrawRect(self, circle, color):
        if np.isnan(color[0]):
            self.printerDraws[0].rectangle(circle, width=3, fill=(0,0,0,25),outline=(0,0,0,255))
            return
        color = [int(c*255) for c in color]
        colors = self.channelPrinterMap(color)
        print(f'colors= {colors}')
        for i, draw in enumerate(self.printerDraws):
            draw.rectangle(circle,width=0,fill=colors[i])
    
    def ExportImages(self,name,pageName):
        print(f"printing {name}/{pageName}")
        for i, im in enumerate(self.printerImages):
            imName = i
            if self.printNames != None:
                imName = self.printNames[i]
            os.makedirs(f'{self.directory}/{name}',exist_ok=True)
            im.save(f'{self.directory}/{name}/print{imName}_page{pageName}.tif')

class Printer():
    def testFunc():
        return "hello World"

    # a printer map takes a color vector and returns an ordered list of CMYK values to print
    def SIGGRAPH2024printerMap(c):
        return [(c[0],c[1],c[2],0),
                (c[3],c[4],0,0),
                (0,0,c[5],0)]
    siggraph2024PrinterNames = ["CMY","IJ","K"]

    def SixInkPrinterMap(c):
        return [(0,c[0],0,0),
                (c[1],0,0,0),
                (0,c[2],c[3],0),
                (0,c[4],c[5],0)]

    sixInkPrinterNames = ["V","T","RC","BM"] # violet turquise red citrus blue maskara

    def __init__(self,printerMap=SIGGRAPH2024printerMap,printerNames=siggraph2024PrinterNames,directory = ".", num_samples = 100, dotSizes = [16,22,28]):
        out = f'{directory}/out'
        os.makedirs(out,exist_ok=True)
        self.directory = directory
        self.outDir = out
        self.num_samples = num_samples
        self.dotSizes = dotSizes
        self.printerMap=printerMap
        self.printerNames = printerNames

    def generate_CC_dots(self,secret,seed=0,l=None):
        size = 1024
        # if not os.path.exists(f'{outdir}/circles.cpkl'):
        def generate_circles():
            # for CC
            np.random.seed(seed)
            radii = []
            for s in range(2000):
                for r in self.dotSizes:
                    radii.append(r)

            np.random.shuffle(radii)
            circles = pc.pack(radii)

            center = size // 2
            l = []

            for i, (x,y,radius) in enumerate(circles):
                # print (f'Processing {i} / 8000 dots...')

                if np.sqrt((x - center) ** 2 + (y - center) ** 2) < center * 0.95:
                    r = radius - np.random.randint(2, 5)
                    print(r)
                    l.append([x,y,r])
            return l
        if l == None:
            l = generate_circles()

        im = Image.open(f'{self.directory}/secretImages/{secret}.png').resize([size,size])

        # given image, inside / outside means numbers / circle background
        image = np.asarray(im)
        outside = np.int32(np.sum(image == 255, -1) == 4)
        inside  = np.int32((image[:,:,3] == 255)) - outside

        # radius of each circle
        rs = []
        inside_values_for_all_circles = []
        outside_values_for_all_circles = []

        # For each circle, check what percentage of each circle overlaps with inside or outside.
        # Here we take N point samples within each circle.
        for i, [x,y,r] in enumerate(l):
            print(r)
            x, y = int(round(x)), int(round(y))
            inside_values = 0
            outside_values = 0

            for _ in range(self.num_samples):
                while True:
                    dx = np.random.uniform(-r, r)
                    dy = np.random.uniform(-r, r)
                    if np.sqrt(dx**2 + dy**2) <= r:
                        break

                if inside[int(np.clip(np.round(y+dy), 0, size-1)), int(np.clip(np.round(x+dx), 0, size-1))]:
                    inside_values += 1
                elif outside[int(np.clip(np.round(y+dy), 0, size-1)), int(np.clip(np.round(x+dx), 0, size-1))]:
                    outside_values += 1

            inside_values_for_all_circles.append(inside_values)
            outside_values_for_all_circles.append(outside_values)
            rs.append(r)

        return [inside_values_for_all_circles, outside_values_for_all_circles, rs, l]

    def generate_CC(self,outside, inside, var, secret, trial,noise = 0,gradient=False):
        [inside_values_for_all_circles, outside_values_for_all_circles, rs, l] = self.generate_CC_dots(secret)
        size = 1024
        n = [np.random.random() for _ in range(len(l))]
        name = f'{var}_{secret}'

        printerMapper = PrinterMapper(self.printerMap,(size,size),self.printerNames,self.outDir)
        
        for i, [x,y,r] in enumerate(l):
            # print(r)
            inside_values = inside_values_for_all_circles[i]
            outside_values = outside_values_for_all_circles[i]
            r = rs[i]
            v  = np.clip(inside_values / self.num_samples * (1 - (n[i] * noise / 100)), 0, 1)
            v2 = np.clip(outside_values/ self.num_samples * (1 - (n[i] * noise / 100)), 0, 1)

            if not gradient:
                v = 1 if v > 0.5 else 0
                v2 = 1 - v

            c = np.asarray(inside) * v + np.asarray(outside) * v2

            printerMapper.DrawEllipse([x-r, y-r, x+r, y+r],c)

        printerMapper.ExportImages(f'{trial}_{name}_noise_{noise}',0)

    def generate_CC_2Image(self,A, B, C, D, var, mainSecret, DistractionSecret, trial,l,noise=0):
        size = 1024
        n = [np.random.random() for _ in range(len(l))]
        name = f'{var}_{mainSecret}_{DistractionSecret}'

        # for (inside,outside,im_name) in [(B,A,im_nameMain),(D,C,im_nameDistraction)]:

        [inside_values_for_all_circles_main, outside_values_for_all_circles_main, rs, l] = self.generate_CC_dots(mainSecret)
        [inside_values_for_all_circles_Distract, outside_values_for_all_circles_Distract, rs, l] = self.generate_CC_dots(DistractionSecret)

        printerMapper = PrinterMapper(self.printerMap,size,self.printerNames,self.outDir)
        
        for i, [x,y,r] in enumerate(l):
            # print(r)
            inside_values_main = inside_values_for_all_circles_main[i]
            outside_values_main = outside_values_for_all_circles_main[i]
            inside_values_dist = inside_values_for_all_circles_Distract[i]
            outside_values_dist = outside_values_for_all_circles_Distract[i]
            r = rs[i]

            v  = np.clip(inside_values_main / self.num_samples * (1 - (n[i] * noise / 100)), 0, 1)
            v2 = np.clip(outside_values_main / self.num_samples * (1 - (n[i] * noise / 100)), 0, 1) 
            v = 1 if v > 0.5 else 0
            v2 = 1 - v

            h  = np.clip(inside_values_dist / self.num_samples * (1 - (n[i] * noise / 100)), 0, 1)
            h2 = np.clip(outside_values_dist / self.num_samples * (1 - (n[i] * noise / 100)), 0, 1) 
            h = 1 if h > 0.5 else 0
            h2 = 1 - h
            
            c = self.lerp2d(A,B,C,D,v,v2,h,h2)

            printerMapper.DrawEllipse([x-r, y-r, x+r, y+r],c)

        printerMapper.ExportImages(f'{trial}_{name}_noise_{noise}',0)
                
    #  a ---x--- b
    #  |
    #  y
    #  |
    #  c ------- d
    # typically, x=1-xi, y=1-yi
    def lerp2d(self,a,b,c,d,x,xi,y,yi):
        print("x=",x)
        print("xi=",xi)
        print("y=",y)
        print("yi=",yi)
        return y*(x*a + xi*b) + yi*(x*c + xi*d)

    def CIJKtoCMYIJK(self,c):
        return [c[0]] + [0,0] + c[1:]

    def div100(self,list):
        return [x/100 for x in list]

    def toList(self,list):
        return [c.tolist() for c in list]

    def depair(self,list):
        out = []
        for pair in list:
            out += [pair[0],pair[1]]
        return out

    def CPVY2CMYIJK(self,cpvy):
        return (cpvy[0],0,0,cpvy[1],cpvy[2],cpvy[3])

    def generateMultiPageDotList(self,colors,name="custom",manualDotSize=None,customLabels=None,rect=False,maxX=None,biggestDots=False,xB=0,yB=0,perPage=None):
        fullLength = len(colors)
        splitColorList = [colors[i*perPage:(i+1)*perPage] for i in range((fullLength//perPage)+1)]
        print("!")
        processed = 0
        for pageNum, colorList in enumerate(splitColorList):
            print("!!")
            if customLabels:
                labels = customLabels[processed:processed+len(colorList)]
            else:
                labels = None
            self.generateDotList(colorList,name,manualDotSize,labels,rect,maxX,biggestDots,xB,yB,pageNum)
            processed += len(colorList)

    def generateDotList(self,colors,name="custom",manualDotSize=None,customLabels=None,rect=False,maxX=None,biggestDots=False,xB=0,yB=0,pageName=""):
        size = 1024
        dotSize = 95
        r = dotSize//2
        border = 20
        if maxX == None:
            maxX = int((size-(4*border))/(dotSize + border))
            maxX -= maxX%2
        elif biggestDots:
            dotSize = (size//maxX)-border
            r = dotSize/2
        if manualDotSize:
            dotSize = manualDotSize
            maxX = int((size-(4*border))/(dotSize + border))
        printerMapper = PrinterMapper(self.printerMap,(int(size*1.05), int(size*1.35)),self.printerNames,self.outDir)
        for i in range(len(colors)):
            x_mini = i%maxX
            y_mini = i//maxX
            x = xB + r + x_mini * (dotSize + border) + 2 * border
            y = yB + r + y_mini * (dotSize + border + 30) + 2 * border
            # y = yB + r + y_mini * (dotSize + border) + 2 * border
            c = colors[i]
            print(c)
            text_placement_x = x - r + 15
            text_placement_y = y + r + border / 2 - 5
            
            if customLabels:
                printerMapper.DrawText(customLabels[i],text_placement_x,text_placement_y)
            else:
                printerMapper.DrawLabel(c,text_placement_x,text_placement_y)

            circle = [x - r, y - r, x + r, y + r]
            if rect:
                printerMapper.DrawRect(circle,c)
            else:
                printerMapper.DrawEllipse(circle,c)
    
        printerMapper.ExportImages(name,pageName)

    #TODO: link this with the cubemap functions, old code:
    # file_path = "/Users/frackfrick/Desktop/SchoolStuff/compColor/25lamy_percentages.npy"
    # sphere_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/spherical_coordinates_of_cubemap.npy"
    # index_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/9x9_idx_into_cubemap.npy"
    # # irgb_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/Archive6/ideal_2.999849055699836_9x9samplesRGBs.npy"
    # irgb_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/Archive7/measured_2.999849055699836_9x9samplesRGBs.npy"
    # # irgb_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/Archive8/ideal_2.999849055699836_9x9samplesRGBs.npy"
    # reflect_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/Archive7/measured_2.999849055699836_9x9sampleSpectra.npy"
    # # reflect_path = "/Users/frackfrick/Desktop/SchoolStuff/Research/Archive8/ideal_2.999849055699836_9x9sampleSpectra.npy"
    # data = np.load(file_path)
    # sphere = np.load(sphere_path)
    # index = np.load(index_path)
    # irgb = np.load(irgb_path)
    # reflectance = np.load(reflect_path)
    def new9x9Grid(self,index,data,name):
        dotSize = 30
        r = dotSize/2
        border = 10
        size = 1024
        printerMapper = PrinterMapper(self.printerMap,(int(size*1.3), size),self.printerNames,self.outDir)
        for square in range(6):
            for c in range(81):
                ind = index[square][c//9][c%9]
                x = r + ind[0] * 8 * (dotSize + border) + 2 * border
                y = r + ind[1] * 8 * (dotSize + border) + 2 * border
                y = size - y
                inde = c
                c = data[square][c]
                # print(inde,c[0])
                circle = [x - r, y - r, x + r, y + r]
                # print("circle is",circle)
                printerMapper.DrawEllipse(circle,c)

        printerMapper.ExportImages(name,0)

    def newGeneralGrid(self,index,data,name,dotSize=30,border=10,labels=False,backupData=None):
        r = dotSize/2
        size = 1024
        sideLen = len(index[0])
        printerMapper = PrinterMapper(self.printerMap,(int(size*1.3), size),self.printerNames,self.outDir)
        for square in range(6):
            for c in range(sideLen**2):
                ind = index[square][c//sideLen][c%sideLen]
                x = r + ind[0] * (sideLen-1) * (dotSize + border) + 2 * border
                y = r + ind[1] * (sideLen-1) * (dotSize + border) + 2 * border
                y = size - y
                inde = c
                c = data[square][c]
                if np.isnan(c[0]) and backupData is not None:
                    c = backupData[square][inde]
                # print(inde,c[0])
                circle = [x - r, y - r, x + r, y + r]
                # print("circle is",circle)
                if not labels:
                    printerMapper.DrawEllipse(circle,c)
                else:
                    printerMapper.DrawText(f"({square},{inde})",x,y)

        printerMapper.ExportImages(name,0)

    def new5x5Grid(self,index,data,name,doRect=False):
        dotSize = 55
        r = dotSize/2
        border = 5
        size = 1024
        printerMapper = PrinterMapper(self.printerMap,(int(size*1.3), size),self.printerNames,self.outDir)
        for square in range(6):
            for c in range(25):
                row = (c//5)*2
                col = (c%5)*2
                c = 9*row + col
                # ind = index[square][(c//5)*2][(c%5)*2]
                ind = index[square][c//9][c%9]
                x = r + ind[0] * 8 * (dotSize/2 + border) + 2 * border
                y = r + ind[1] * 8 * (dotSize/2 + border) + 2 * border
                y = size - y
                inde = c
                c = data[square][c]
                # print(inde,c[0])
                circle = [x - r, y - r, x + r, y + r]
                # print("circle is",circle)
                if doRect:
                    printerMapper.DrawRect(circle,c)
                else:
                    printerMapper.DrawEllipse(circle,c)

        printerMapper.ExportImages(name,0)

    def new9x9Shuffle(self,index,data,swapIndicies,name,backupData=None):
        dotSize = 55
        r = dotSize/2
        border = 15
        size = 1024
        printerMapper = PrinterMapper(self.printerMap,(int(size*1.3), size),self.printerNames,self.outDir)
        for square in (0,5):
            for c in range(81):
                if square == 5:
                    ind = index[square][c//9][8-(c%9)]
                else:
                    ind = index[square][c//9][c%9]
                x = 20 + r + ind[0] * 8 * (dotSize + border) + 2 * border
                y = r + ind[1] * 8 * (dotSize + border) + 2 * border
                y = size - 20 - y
                if square == 0:
                    y += 920
                    x -= 600
                if square == 5:
                    y -= 200
                    x += 50
                inde = c
                if c in swapIndicies:
                    c = data[5-square][9*(c//9) + 8 - (c%9)]
                    if np.isnan(c[0]) and backupData is not None:
                        # print(backupData)
                        c = backupData[5-square][9*(inde//9) + 8 - (inde%9)]
                else:
                    c = data[square][c]
                    if np.isnan(c[0]) and backupData is not None:
                        c = backupData[square][inde]
                # print(inde,c[0])
                circle = [x - r, y - r, x + r, y + r]
                # print("circle is",circle)
                printerMapper.DrawEllipse(circle,c)

        printerMapper.ExportImages(name,0)

    def newRGBGrid(self,index,irgb):
        size = 1024
        dotSize = 20
        r = dotSize/2
        border = 20
        im_CMY = Image.new('RGB', (int(size*1.2), size),color=(255,255,255,255))
        dr_CMY = ImageDraw.Draw(im_CMY)
        part = 0
        total = 0
        for square in range(6):
            for c in range(81):
                total += 1
                ind = index[square][c//9][c%9]
                x = 0 + r + ind[0] * 8 * (dotSize + border) + 2 * border
                y = -100 + r + ind[1] * 8 * (dotSize + border) + 2 * border
                y = 2300 - y
                if square == 0:
                    y -= (dotSize+border)*2
                    x += (dotSize+border)*1
                if square == 1:
                    y -= (dotSize+border)*1
                if square == 2:
                    x += (dotSize+border)*1
                    y -= (dotSize+border)*1
                if square == 3:
                    x += (dotSize+border)*2
                    y -= (dotSize+border)*1
                if square == 4:
                    x += (dotSize+border)*3
                    y -= (dotSize+border)*1
                if square == 5:
                    x += (dotSize+border)*1
                    # y -= 100
                
                inde = c
                c = irgb[square*81 + c]
                c = (c[0],c[1],c[2])
                # print(c[0])
                print(c)
                w = 0
                
                # (0.9028771664209356, 0.9031003998921243, 0.9030793989690995)
                if (abs(c[0] - 0.9028771664209356) < 0.01 and abs(c[1] - 0.9031003998921243) < 0.01 and abs(c[0] - 0.903079398969099) < 0.01):
                    print("ww")
                    c = (.9,.9,.9)
                    w = 3
                    part += 1
                else:
                    mul = 1.15
                    c = (c[0]*mul,c[1]*mul,c[2]*mul)
                    # c = c
                # if (square == chosen[0] and inde == chosen[1]):
                #     c = (1,0,1)
                circle = [x - r, y - r, x + r, y + r]
                _cmy = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255), 255)
                dr_CMY.ellipse(circle, width=w, fill=_cmy,outline=(0,0,0,255))
        im_CMY.save(f'./SuperNewcustomRGB.png')

# default = ImageFont.load_default()
# font = default


# size = 1024
# dotSize = 75
# border = 20
# # size = 260
# # dotSize = 75
# # border = 10
# r = dotSize / 2

# gridSize = r + 9 * (dotSize + border) + 2 * border

p = Printer(directory="./chromalab")

# testDict = {
#     "A": ([25,32,47,85],[37,90,2,50],'87'), #downstairs
#     "B": ([27,2,57,85],[42,95,2,17],'85'),
#     "C": ([30,22,50,90],[42,95,0,52],'73'),
#     "D": ([30,25,47,90],[37,90,2,50],'27'), #2850k
#     "E": ([27,15,52,82],[40,87,0,25],'39'),
#     "F": ([30,7,52,95],[40,97,0,50],'68'),
#     "G": ([30,17,50,92],[37,90,2,62],'96'), #4800k
#     "H": ([30,17,50,92],[37,92,2,65],'72'),
#     "I": ([32,17,50,80],[40,87,0,25],'35'),
#     "J": ([30,17,50,92],[37,90,2,62],'67'), #6500k
#     "K": ([30,22,50,90],[37,90,2,50],'64'),
#     "L": ([27,7,55,85],[40,87,0,45],'89'),
# }

# for t in testDict.keys():
#     inside,outside,secret = testDict[t]
#     print(inside)
#     inside = p.CIJKtoCMYIJK(p.div100(inside))
#     outside = p.CIJKtoCMYIJK(p.div100(outside))
#     p.generate_CC(inside,outside,0,secret,t,True)

# l=[]
# for c in range(2):
#     for m in range(2):
#         for y in range(2):
#             for i in range(2):
#                 for j in range(2):
#                     for k in range(2):
#                         l += [[c,m,y,i,j,k]]
# p.generateDotList(l)

