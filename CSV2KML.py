#!/usr/bin/env python2.7
"""
=============== Converts a CSV to KML ===============
Created on Mon Jul 06 12:18:06 2015
@author: Trent Jansen-Sturgeon

"""

import os

from astropy.table import Table
import numpy as np


def fetchTriangdata(ifile):
    triangulation_table = Table.read(ifile, format='ascii.csv', guess=False, delimiter=',')

    Long0 = triangulation_table.meta['p_lat']
    Lat0 = triangulation_table.meta['p_long']
    H0 = triangulation_table.meta['p_height']
    CamName = triangulation_table.meta['telescope'] + " " + triangulation_table.meta['location']

    return triangulation_table, Long0, Lat0, H0, CamName


def Path(FileName):

    Data, Long0, Lat0, H0, CamName = fetchTriangdata(FileName)

    #from numpy import genfromtxt as gft

    # Extract the camera data
    # Data=gft(FileName,names=True,delimiter=',',skip_header=1,dtype=None)

    # Open the file to be written.
    outputname = os.path.join(os.path.dirname(FileName), os.path.basename(FileName).split('.')[0] + '_path.kml')
    f = open(outputname, 'w')

    # Writing the kml file.
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
    f.write('<Document>\n')
    f.write('	<name>' + FileName.split('/')[-1] + '_path' + '</name>\n')
    f.write('	<StyleMap id="msn_placemark_circle0">\n')
    f.write('       <Pair>\n')
    f.write('           <key>normal</key>\n')
    f.write('           <styleUrl>#sn_placemark_circle0</styleUrl>\n')
    f.write('       </Pair>\n')
    f.write('       <Pair>\n')
    f.write('           <key>highlight</key>\n')
    f.write('           <styleUrl>#sh_placemark_circle_highlight0</styleUrl>\n')
    f.write('       </Pair>\n')
    f.write('   </StyleMap>\n')
    f.write('   <Style id="sn_placemark_circle0">\n')
    f.write('       <IconStyle>\n')
    f.write('           <color>ff0000ff</color>\n')
    f.write('           <scale>0.5</scale>\n')
    f.write('           <Icon>\n')
    f.write('               <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
    f.write('           </Icon>\n')
    f.write('       </IconStyle>\n')
    f.write('   </Style>\n')
    f.write('   <Style id="sh_placemark_circle_highlight0">\n')
    f.write('       <IconStyle>\n')
    f.write('           <color>ff0000ff</color>\n')
    f.write('           <scale>0.5</scale>\n')
    f.write('           <Icon>\n')
    f.write('               <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href>\n')
    f.write('           </Icon>\n')
    f.write('       </IconStyle>\n')
    f.write('   </Style>\n')
    f.write('	<Placemark>\n')
    #f.write('		<name>' + FileName.split('.')[0] + '</name>\n')
    f.write('		<open>1</open>\n')
    f.write('		<styleUrl>#msn_placemark_circle0</styleUrl>\n')
    f.write('		<LineString>\n')
    f.write('			<extrude>1</extrude>\n')
    f.write('			<tessellate>1</tessellate>\n')
    f.write('			<altitudeMode>absolute</altitudeMode>\n')
    f.write('			<coordinates>\n')
    # print Data
    for row in Data:
        f.write('					' + str(row['longitude']) + ',' +
                str(row['latitude']) + ',' + str(row['height']) + '\n')
    f.write('			</coordinates>\n')
    f.write('		</LineString>\n')
    f.write('	</Placemark>\n')

    f.write('</Document>\n')
    f.write('</kml>\n')
    f.close()
    print('File Created: ' + outputname)


def Points(FileName):

    #from numpy import genfromtxt as gft

    # Extract the camera data
    # Data=gft(FileName,names=True,delimiter=',',skip_header=1,dtype=None)

    #Data, Long0, Lat0, H0, CamName = fetchTriangdata(FileName)
    Data = fetchTriangdata(FileName)

    # Open the file to be written.
    outputname = os.path.join(os.path.dirname(FileName), os.path.basename(FileName).split('.')[0] + '_points.kml')
    f = open(outputname, 'w')

    # Writing the kml file.
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://www.opengis.net/kml/2.2" xmlns:gx="http://www.google.com/kml/ext/2.2" xmlns:kml="http://www.opengis.net/kml/2.2" xmlns:atom="http://www.w3.org/2005/Atom">\n')
    f.write('<Document>\n')
    f.write('   <name>' + FileName.split('/')[-1] + '_path' + '</name>\n')
    f.write('   <StyleMap id="msn_placemark_circle0">\n')
    f.write('       <Pair>\n')
    f.write('           <key>normal</key>\n')
    f.write('           <styleUrl>#sn_placemark_circle0</styleUrl>\n')
    f.write('       </Pair>\n')
    f.write('       <Pair>\n')
    f.write('           <key>highlight</key>\n')
    f.write('           <styleUrl>#sh_placemark_circle_highlight0</styleUrl>\n')
    f.write('       </Pair>\n')
    f.write('   </StyleMap>\n')
    f.write('   <Style id="sn_placemark_circle0">\n')
    f.write('       <IconStyle>\n')
    f.write('           <color>ff0000ff</color>\n')
    f.write('           <scale>0.5</scale>\n')
    f.write('           <Icon>\n')
    f.write('               <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href>\n')
    f.write('           </Icon>\n')
    f.write('       </IconStyle>\n')
    f.write('   </Style>\n')
    f.write('   <Style id="sh_placemark_circle_highlight0">\n')
    f.write('       <IconStyle>\n')
    f.write('           <color>ff0000ff</color>\n')
    f.write('           <scale>0.5</scale>\n')
    f.write('           <Icon>\n')
    f.write('               <href>http://maps.google.com/mapfiles/kml/shapes/placemark_circle_highlight.png</href>\n')
    f.write('           </Icon>\n')
    f.write('       </IconStyle>\n')
    f.write('   </Style>\n')

    PtNum = 0
    for row in Data:
        PtNum += 1
        f.write('		<Placemark>\n')
        #f.write('			<name>' + str(PtNum) + '</name>\n')
        f.write('			<description>' + str(row['time']) + '</description>\n')
        f.write('			<styleUrl>#msn_placemark_circle0</styleUrl>\n')
        f.write('			<Point>\n')
        f.write('				<altitudeMode>absolute</altitudeMode>\n')
        f.write('				<coordinates>' + str(row['p_long']) + "," +
                str(row['p_lat']) + "," + str(row['p_height']) + '</coordinates>\n')
        f.write('			</Point>\n')
        f.write('		</Placemark>\n')

    f.write('</Document>\n')
    f.write('</kml>\n')
    f.close()
    print('File Created: ' + outputname)


def Projection(FileName, Colour='33ff0000'):

    #from csv import reader
    #from numpy import genfromtxt as gft

    # Extract the camera's name and position
    # CamLocation=next(reader(open(FileName,'r')))
    # CamName=str(CamLocation[0])
    # Lat0=float(CamLocation[2]) # Latitude [deg]
    # Long0=float(CamLocation[1]) # Longitude [deg]
    # H0=float(CamLocation[3]) # Height [m]

    #Data, Long0, Lat0, H0, CamName = fetchTriangdata(FileName)
    Data = fetchTriangdata(FileName)

    # Extract the camera data
    # Data=gft(FileName,names=True,delimiter=',',skip_header=1,dtype=None)

    # Open the file to be written.
    outputname = os.path.join(os.path.dirname(FileName), os.path.basename(FileName).split('.')[0] + '_camera.kml')
    f = open(outputname, 'w')

    f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    f.write('<kml xmlns="http://earth.google.com/kml/2.1">\n')
    f.write('<Document>\n')
    #f.write('<name>' + FileName.split('.')[0] + '_camera' + '</name>\n')
    f.write('<open>1</open>\n')
    f.write('<Placemark>\n')
    f.write('	<Style id="camera">\n')
    f.write('		<LineStyle>\n')
    f.write('			<width>1</width>\n')
    f.write('		</LineStyle>\n')
    f.write('		<PolyStyle>\n')
    f.write('			<color>' + str(Colour) + '</color>\n')
    f.write('		</PolyStyle>\n')
    f.write('	</Style>\n')
    f.write('	<styleUrl>#camera</styleUrl>\n')
    f.write('<name>' + str(CamName) + '</name>\n')
    f.write('<Polygon>\n')
    f.write('	<extrude>0</extrude>\n')
    f.write('	<altitudeMode>absolute</altitudeMode>\n')
    f.write('	<outerBoundaryIs>\n')
    f.write('		<LinearRing>\n')
    f.write('		<coordinates>\n')
    f.write('			' + str(Long0) + ',' + str(Lat0) + ',' + str(H0) + '\n')
    # for row in [Data[0],Data[-1]]:
    for row in Data:
        f.write('				' + str(row['p_long']) + "," +
                str(row['p_lat']) + "," + str(row['p_height']) + '\n')
    f.write('			' + str(Long0) + ',' + str(Lat0) + ',' + str(H0) + '\n')
    f.write('		</coordinates>\n')
    f.write('		</LinearRing>\n')
    f.write('	</outerBoundaryIs>\n')
    f.write('</Polygon>\n')
    f.write('</Placemark>\n')
    f.write('</Document>\n')
    f.write('</kml>\n')

    f.close()
    print('File Created: ' + outputname)


if __name__ == "__main__":
    import sys
    filename = sys.argv[1]
    print(filename)
    #Projection(filename)
    #Path(filename)
    Points(filename)
