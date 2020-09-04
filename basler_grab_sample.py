from pypylon import pylon
import cv2
from sampledetection import SampleDetection

coords = [207, 173]
offset = [0, 0]
radius = [143]

#threshold = [20, 16, 120]
#threshold = [115, 16, 152]
threshold = [70, 10, 70]

# [207, 173] [143] [70, 10, 70] has_sample? True 976
#     pylon.FeaturePersistence_Load('D:/images/certa/c4.1/PoC/20200214/1636.pfs'.format(name),

converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# demonstrate some feature access
new_width = camera.Width.GetValue() - camera.Width.GetInc()
if new_width >= camera.Width.GetMin():
    camera.Width.SetValue(new_width)

numberOfImagesToGrab = 10000
# camera.StartGrabbingMax(numberOfImagesToGrab)
camera.StartGrabbing()



def FeaturePersistence_Load(name):
    camera.StopGrabbing()
    pylon.FeaturePersistence_Load('D:/images/certa/c4.1/PoC/20200214/{}.pfs'.format(name),
                                  camera.GetNodeMap())
    camera.StartGrabbing()


frames = 1

# FeaturePersistence_Load('Image__2020-02-06__16-51-00-fixed')
FeaturePersistence_Load('1636')
saturation = camera.BslSaturationValueRaw.GetValue()

contrast = 0
brightness = 0

while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        image = converter.Convert(grabResult)
        img = image.GetArray()
        cv2.imshow('IN', img)

        sd = SampleDetection(img, debug=True)
        sd.crop(coords[0], coords[1], radius[0])
        # sd.auto_crop(radius[0], debug=True)

        # ['FI', [rangeH[0], rangeS[0], rangeV[0]], [rangeH[1], rangeS[1], rangeV[1]], threshold[2]],

        # TODO: review DR1, DR2 color ranges (getting background noise)
        filters = [
            ['DR1', [39, 214, 0], [55, 255, 255], threshold[0]],
            ['DR2', [30, 210, 40], [55, 256, 100], threshold[0]],
            ['DR3', [0, 60, 5], [19, 255, 145], threshold[0]],
        ]
        debug = False
        for filter in filters:
            sd.filter(filter[0], [filter[1], filter[2]], threshold[0], debug)
            sd.transform(filter[0], filter[3], debug)

        # merge group of colours
        sd.merge('DR', [0, 1, 2], True)

        # TODO: review LR1, LR2 color ranges (too open, getting OR ranges)
        filters = [
            ['LR1', [0, 71, 0], [60, 255, 255], threshold[1]],
            ['LR2', [0, 50, 100], [39, 105, 230], threshold[1]],
        ]
        for filter in filters:
            sd.filter(filter[0], [filter[1], filter[2]], threshold[1], debug)
            sd.transform(filter[0], filter[3], debug)

        # merge group of colours
        sd.merge('LR', [3, 4], True)

        filters = [
            ['OR1', [0, 85, 185], [50, 190, 255], threshold[1]],
            ['OR2', [10, 100, 146], [25, 190, 255], threshold[1]],
            ['OR3', [10, 100, 146], [30, 255, 255], threshold[1]],
        ]
        for filter in filters:
            sd.filter(filter[0], [filter[1], filter[2]], threshold[1], debug)
            sd.transform(filter[0], filter[3], debug)

        # merge group of colours
        sd.merge('OR', [5, 6, 7], True)

        filters = [
            ['WT1', [0, 0, 180], [65, 70, 220], threshold[2]],
            ['WT2', [0, 0, 205], [33, 50, 255], threshold[2]],
            ['WT3', [15, 55, 172], [28, 90, 204], threshold[2]],
            ['WT4', [10, 35, 160], [28, 75, 218], threshold[2]],
            ['WT5', [140, 0, 245], [160, 10, 255], threshold[2]],
            ['WT6', [120, 10, 130], [140, 65, 195], threshold[2]],
            ['WT7', [135, 118, 119], [178, 142, 160], threshold[2]],
            ['WT8', [140, 124, 139], [165, 143, 159], threshold[2]],
        ]
        for filter in filters:
            # debug = False
            # if filter[0] == 'WT4':
            #    debug = True
            sd.filter(filter[0], [filter[1], filter[2]], threshold[2], debug)
            sd.transform(filter[0], filter[3], debug)

        # merge group of colours
        sd.merge('WT', [8, 9, 10, 11, 12, 13, 14, 15], True)

        # put all filtered pixels into one image
        sd.merge_all()

        cv2.imshow('OUT', sd.output)
        key = cv2.waitKey(1)

        # print('.', end='')
        # if frames % 100 == 0:
        #    print(frames)
        frames += 1
        # print(key)

        # 1
        if key == 49:
            FeaturePersistence_Load('Image__2020-02-06__16-51-00-fixed')
        # 2
        if key == 50:
            FeaturePersistence_Load('Image__2020-02-06__16-51-56')
        # 3
        if key == 51:
            FeaturePersistence_Load('Image__2020-02-06__16-52-33')
        # 4
        if key == 52:
            FeaturePersistence_Load('Image__2020-02-06__16-53-21')
        if key == 53:
            FeaturePersistence_Load('Image__2020-02-06__16-57-35')
        if key == 54:
            FeaturePersistence_Load('Image__2020-02-06__16-58-00')
        if key == 55:
            FeaturePersistence_Load('Image__2020-02-06__16-58-26')

        # S
        if key == 115:
            saturation += 10
            camera.BslSaturationValueRaw.SetValue(saturation)
        # D
        if key == 100:
            saturation -= 10
            camera.BslSaturationValueRaw.SetValue(saturation)
        # C
        if key == 99:
            contrast += 1
            camera.BslContrastRaw.SetValue(contrast)
        # V
        if key == 118:
            contrast -= 1
            camera.BslContrastRaw.SetValue(contrast)
        # B
        if key == 98:
            brightness += 1
            camera.BslBrightnessRaw.SetValue(brightness)
        # N
        if key == 110:
            brightness -= 1
            camera.BslBrightnessRaw.SetValue(brightness)

        # H
        # if key == 104:
        #    print(camera.BlsHueRaw.GetValue())
        #    # camera.BlsHueRaw.SetValue(hue)

        # I
        if key == 105:
            coords[1] = coords[1] + 1
        # M
        if key == 109:
            coords[1] = coords[1] - 1
        # J
        if key == 106:
            coords[0] = coords[0] + 1
        # K
        if key == 107:
            coords[0] = coords[0] - 1

        # +
        if key == 43:
            radius[0] = radius[0] + 1
        # -
        if key == 45:
            radius[0] = radius[0] - 1

        # O
        if key == 111:
            # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('{} {} {} has_sample? {} {}'.format(coords, radius, threshold, sd.has_sample(300), sd.density()))
            # cv2.imwrite('{}/{} {} {} {}.jpg'.format(path_out, timestamp, rangeH, rangeS, rangeV), sd.merged)

        if key == 27:
            cv2.destroyAllWindows()
            exit(0)

    grabResult.Release()

camera.Close()

