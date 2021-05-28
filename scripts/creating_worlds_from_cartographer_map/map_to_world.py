#!/usr/bin/env python

import os 
import argparse
import numpy as np 
import cv2 
import yaml
from lxml import etree as ET
from copy import copy
import sys
import json
import math



class WorldGenerator(object):

    kernel_size = 4 # for 2D map image preprocessing

    LOOKUP_TABLE = np.array([
    1.0,
    1.0,
    2.0,
    6.0,
    24.0,
    120.0,
    720.0,
    5040.0,
    40320.0,
    362880.0,
    3628800.0,
    39916800.0,
    479001600.0,
    6227020800.0,
    87178291200.0,
    1307674368000.0,
    20922789888000.0,
    355687428096000.0,
    6402373705728000.0,
    121645100408832000.0,
    2432902008176640000.0,
    51090942171709440000.0,
    1124000727777607680000.0,
    25852016738884976640000.0,
    620448401733239439360000.0,
    15511210043330985984000000.0,
    403291461126605635584000000.0,
    10888869450418352160768000000.0,
    304888344611713860501504000000.0,
    8841761993739701954543616000000.0,
    265252859812191058636308480000000.0,
    8222838654177922817725562880000000.0,
    263130836933693530167218012160000000.0
     ])

    def Ni(self, n, i):
        a1=self.LOOKUP_TABLE[n]
        a2=self.LOOKUP_TABLE[i]
        a3=self.LOOKUP_TABLE[n-i]
        ni=a1/(a2*a3)
        return ni

    def Bernstein(self,n, i, t):
        if t == 0.0 and i == 0:
            ti = 1.0
        else:
            ti = t**i

        if (n == i and t == 1.0):
            tni = 1.0
        else:
            tni = (1-t)**(n-i)

        basis = self.Ni(n, i)*ti*tni
        return basis

    def Bezier2D(self, trajectory, segment):
        cpts = segment['Parts']
        npts = len(trajectory)
        interpolatedPoints = [0]*cpts
        icount = 0
        t = 0.0
        step = 1.0/(cpts-1)

        for i in range(cpts):
            if (1.0-t) < 5e-6:
                t = 1.0

            jcount = 0
            interpolatedPoints[icount] = [0.0, 0.0]

            for j in range(npts):
                basis = self.Bernstein(npts - 1, j, t)
                interpolatedPoints[icount][0] += basis*trajectory[jcount][0]
                interpolatedPoints[icount][1] += basis*trajectory[jcount][1]
                jcount += 1

            icount += 1
            t += step
        return interpolatedPoints



    def __init__(self, yaml_path,layout_path):
        self.yaml_path = yaml_path 
        self.pgm_path = os.path.join(os.path.dirname(yaml_path),
                                     os.path.splitext(os.path.basename(yaml_path))[0] + '.pgm')
        self.generated_world_path = os.path.join(os.path.dirname(os.path.dirname(yaml_path)),
                                                 'worlds',
                                                 'generated_' + os.path.splitext(os.path.basename(yaml_path))[0] + '.world')
        print(self.yaml_path)
        print(self.generated_world_path)
        self.layout_path=layout_path  
        print(self.layout_path)                                       

        ####################Prepare the image: ####################################
        ## Converts the image from cartographer to one from gmapping, doesnt really matter what are the values you set for the black white etc, because the map and the world will be
        ## perfectly syncronized
        img=cv2.imread(self.pgm_path)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width= img.shape
        self.width=width
        self.height=height

        for x in range(0,width):
            for y in range(0,height):
                if(img[y,x]<=65):
                    img[y,x]=0
                if(img[y,x]>=196):
                    img[y,x]=254
                if(img[y,x]<196 and img[y,x]>65):
                    img[y,x]=205
        print('Preprocessing image finished...')
        cv2.imwrite('savedImage.pgm', img)
        #############################################################################

        self.map = cv2.imread('savedImage.pgm', -1).astype(np.float)

        self.map=self.map[::-1,:]
        
        self.map[self.map==0] = 255 # obstacle
        self.map[self.map==254] = 0 # empty 
        self.map[self.map==205] = 0 # unknown 

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        self.map = cv2.morphologyEx(self.map, cv2.MORPH_CLOSE, kernel)

        with open(self.yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)

        self.origin = np.array([yaml_data['origin'][0], yaml_data['origin'][1]])
        self.resolution = yaml_data['resolution']






    def write_basics(self):
        sdf = ET.Element("sdf",version="1.6")
        world = ET.SubElement(sdf, "world", name="default")
        # global light source
        include = ET.SubElement(world, 'include')
        uri = ET.SubElement(include, 'uri')
        uri.text = 'model://sun'
        # simple ground plane
        # include = ET.SubElement(world, 'include')
        # uri = ET.SubElement(include, 'uri')
        # uri.text = 'model://ground_plane'
        # ode
        physics = ET.SubElement(world, 'physics', type='ode')
        real_time_update_rate = ET.SubElement(physics, 'real_time_update_rate')
        real_time_update_rate.text = '1000.0'
        max_step_size = ET.SubElement(physics, 'max_step_size')
        max_step_size.text = '0.001'
        real_time_factor = ET.SubElement(physics, 'real_time_factor')
        real_time_factor.text = '1'
        ode = ET.SubElement(physics, 'ode')
        solver = ET.SubElement(ode, 'solver')
        type = ET.SubElement(solver, 'type')
        type.text = 'quick'
        iters = ET.SubElement(solver, 'iters')
        iters.text = '150'
        precon_iters = ET.SubElement(solver, 'precon_iters')
        precon_iters.text = '0'
        sor = ET.SubElement(solver, 'sor')
        sor.text = '1.400000'
        use_dynamic_moi_rescaling = ET.SubElement(solver, 'use_dynamic_moi_rescaling')
        use_dynamic_moi_rescaling.text = '1'
        constraints = ET.SubElement(ode, 'constraints')
        cfm = ET.SubElement(constraints, 'cfm')
        cfm.text = '0.00001'
        erp = ET.SubElement(constraints, 'erp')
        erp.text = '0.2'
        contact_max_correcting_vel = ET.SubElement(constraints, 'contact_max_correcting_vel')
        contact_max_correcting_vel.text = '2000.000000'
        contact_surface_layer = ET.SubElement(constraints, 'contact_surface_layer')
        contact_surface_layer.text = '0.01000'

        # scene
        scene = ET.SubElement(world, 'scene')
        ambient = ET.SubElement(scene, 'ambient')
        ambient.text = '0.4 0.4 0.4 1'
        background = ET.SubElement(scene, 'background')
        background.text = '0.7 0.7 0.7 1'
        shadows = ET.SubElement(scene, 'shadows')
        shadows.text = 'true'

        # gui
        gui = ET.SubElement(world, 'gui', fullscreen='0')
        camera = ET.SubElement(gui, 'camera', name='user_camera')
        pose = ET.SubElement(camera, 'pose')
        pose.text = '0.0 0.0 17.0 -1.5708 1.5708 0'
        view_controller = ET.SubElement(camera, 'view_controller')
        view_controller.text ='orbit'

        return sdf, world






    def generate_v_wall(self, parent, x, y, length, id):
        collision = ET.SubElement(parent, 'collision', name='obstacle_{}'.format(id))
        pose = ET.SubElement(collision, 'pose')
        pose.text = '{} {} 0 0 0 1.5708'.format(str(x), str(y))
        geometry = ET.SubElement(collision, 'geometry')
        box = ET.SubElement(geometry, 'box')
        size = ET.SubElement(box, 'size')
        size.text = '{} 0.15 1.0'.format(str(length))

        visual = ET.SubElement(parent, 'visual', name='obstacle_{}'.format(id))
        pose = ET.SubElement(visual, 'pose')
        pose.text = '{} {} 0 0 0 1.5708'.format(str(x), str(y))
        geometry = ET.SubElement(visual, 'geometry')
        box = ET.SubElement(geometry, 'box')
        size = ET.SubElement(box, 'size')
        size.text = '{} 0.15 1.0'.format(str(length))
        material = ET.SubElement(visual, 'material')
        script = ET.SubElement(material, 'script')
        uri = ET.SubElement(script, 'uri')
        uri.text = 'file://media/materials/scripts/gazebo.material'
        name = ET.SubElement(script, 'name')
        name.text = 'Gazebo/Grey'
        ambient = ET.SubElement(material, 'ambient')
        ambient.text = '1 1 1 1'






    def generate_h_wall(self, parent, x, y, length, id):
        collision = ET.SubElement(parent, 'collision', name='obstacle_{}'.format(id))
        pose = ET.SubElement(collision, 'pose')
        pose.text = '{} {} 0 0 0 0'.format(str(x), str(y))
        geometry = ET.SubElement(collision, 'geometry')
        box = ET.SubElement(geometry, 'box')
        size = ET.SubElement(box, 'size')
        size.text = '{} 0.15 1.0'.format(str(length))

        visual = ET.SubElement(parent, 'visual', name='obstacle_{}'.format(id))
        pose = ET.SubElement(visual, 'pose')
        pose.text = '{} {} 0 0 0 0'.format(str(x), str(y))
        geometry = ET.SubElement(visual, 'geometry')
        box = ET.SubElement(geometry, 'box')
        size = ET.SubElement(box, 'size')
        size.text = '{} 0.15 1.0'.format(str(length))
        # cylinder = ET.SubElement(geometry, 'cylinder')
        material = ET.SubElement(visual, 'material')
        script = ET.SubElement(material, 'script')
        uri = ET.SubElement(script, 'uri')
        uri.text = 'file://media/materials/scripts/gazebo.material'
        name = ET.SubElement(script, 'name')
        name.text = 'Gazebo/Grey'
        ambient = ET.SubElement(material, 'ambient')
        ambient.text = '1 1 1 1'
  






    def is_v_wall_type(self, r, c, gridmap):
        """
            Return True if it is belong to vertical wall ...

            This function helps to reduce the number of walls to generate
        """

        vl = 0 # expected vertical length
        rr = r
        cc = c
        while self.map[rr][cc] == 255:
            rr += 1
            vl += 1
            
        hl = 0 # expected horizontal length
        rr = r
        cc = c
        while self.map[rr][cc] == 255:
            cc += 1
            hl += 1
        return vl > hl


    def generate_layout(self,parent):
        try:
            with open(self.layout_path) as f:
                data=json.load(f)

            trajectory=[]
            stations=[]

            id_line=0
            for segment in data['Segments']:
                if segment['Vectors']:
                    pointStart=[segment['StartVector']['X']/1000.0, segment['StartVector']['Y']/1000.0]
                    pointEnd=[segment['EndVector']['X']/1000.0, segment['EndVector']['Y']/1000.0]
                    trajectory.append(pointStart)
                    for ptinvectors in segment['Vectors']:
                        trajectory.append([ptinvectors['X']/1000.0, ptinvectors['Y']/1000.0])
                    trajectory.append(pointEnd)
                    curve_points=self.Bezier2D(trajectory,segment)
                    trajectory.append(curve_points)
                    print(curve_points)
                    # marker_lines.points.extend(curve_points)

                    for point in curve_points:
                        # point=[station['Vector']['X']/1000.0, station['Vector']['Y']/1000.0]
                        collision = ET.SubElement(parent, 'collision', name='curve_{}'.format(id_line))
                        pose = ET.SubElement(collision, 'pose')
                        pose.text = '{} {} 0 0 0 {}'.format(str(point[0]),str(point[1]) , '0')
                        geometry = ET.SubElement(collision, 'geometry')
                        plane = ET.SubElement(geometry, 'plane')
                        # size = ET.SubElement(plane, 'size')
                        # size.text = '0.05 {}'.format(str(length))
                        surface=ET.SubElement(collision,'surface')
                        friction=ET.SubElement(surface,'friction')
                        ode=ET.SubElement(friction,'ode')
                        mu=ET.SubElement(ode,'mu')
                        mu.text='100'
                        mu2=ET.SubElement(ode,'mu2')
                        mu2.text='50'
                        torsional=ET.SubElement(friction,'torsional')
                        ode2=ET.SubElement(torsional,'ode')
                        contact=ET.SubElement(surface,'contact')
                        ode3=ET.SubElement(contact,'ode')
                        bounce=ET.SubElement(surface,'bounce')
                        max_contacts=ET.SubElement(collision,'max_contacts')
                        max_contacts.text='10'

                        visual = ET.SubElement(parent, 'visual', name='curve_{}'.format(id_line))
                        pose = ET.SubElement(visual, 'pose')
                        pose.text = '{} {} 0 0 0 {}'.format(str(point[0]),str(point[1]) , '0')
                        cast_shadows=ET.SubElement(visual,'cast_shadows')
                        geometry = ET.SubElement(visual, 'geometry')
                        cylinder = ET.SubElement(geometry, 'cylinder')
                        radius = ET.SubElement(cylinder, 'radius')
                        radius.text = '0.02'
                        length=ET.SubElement(cylinder,'length')
                        length.text='0'
                        # cylinder = ET.SubElement(geometry, 'cylinder')
                        material = ET.SubElement(visual, 'material')
                        script = ET.SubElement(material, 'script')
                        uri = ET.SubElement(script, 'uri')
                        uri.text = 'file://media/materials/scripts/gazebo.material'
                        name = ET.SubElement(script, 'name')
                        name.text = 'Gazebo/Red'
                        # ambient = ET.SubElement(material, 'ambient')
                        # ambient.text = '1 1 1 1'
                        id_line+=1
                    del trajectory[:]
                # List of layout points is not according to id
                # layout_list.append([segment['StartVector']['X']/1000, segment['StartVector']['Y']/1000])
                else:
                    pointStart=[segment['StartVector']['X']/1000.0, segment['StartVector']['Y']/1000.0]
                    pointEnd=[segment['EndVector']['X']/1000.0, segment['EndVector']['Y']/1000.0]
                    mean_point=[pointStart[0]+(pointEnd[0]-pointStart[0])/2, pointStart[1]+(pointEnd[1]-pointStart[1])/2]
                    length=math.sqrt((pointEnd[0]-pointStart[0])**2+(pointEnd[1]-pointStart[1])**2)
                    angle=math.atan((pointEnd[1]-pointStart[1])/(pointEnd[0]-pointStart[0]))+1.5708
                    # if segment['Vectors']:
                    collision = ET.SubElement(parent, 'collision', name='line_{}'.format(id_line))
                    pose = ET.SubElement(collision, 'pose')
                    pose.text = '{} {} 0 0 0 {}'.format(str(mean_point[0]),str(mean_point[1]) , str(angle))
                    geometry = ET.SubElement(collision, 'geometry')
                    plane = ET.SubElement(geometry, 'plane')
                    size = ET.SubElement(plane, 'size')
                    size.text = '0.05 {}'.format(str(length))
                    surface=ET.SubElement(collision,'surface')
                    friction=ET.SubElement(surface,'friction')
                    ode=ET.SubElement(friction,'ode')
                    mu=ET.SubElement(ode,'mu')
                    mu.text='100'
                    mu2=ET.SubElement(ode,'mu2')
                    mu2.text='50'
                    torsional=ET.SubElement(friction,'torsional')
                    ode2=ET.SubElement(torsional,'ode')
                    contact=ET.SubElement(surface,'contact')
                    ode3=ET.SubElement(contact,'ode')
                    bounce=ET.SubElement(surface,'bounce')
                    max_contacts=ET.SubElement(collision,'max_contacts')
                    max_contacts.text='10'

                    visual = ET.SubElement(parent, 'visual', name='line_{}'.format(id_line))
                    pose = ET.SubElement(visual, 'pose')
                    pose.text = '{} {} 0 0 0 {}'.format(str(mean_point[0]),str(mean_point[1]) , str(angle))
                    cast_shadows=ET.SubElement(visual,'cast_shadows')
                    geometry = ET.SubElement(visual, 'geometry')
                    plane = ET.SubElement(geometry, 'plane')
                    size = ET.SubElement(plane, 'size')
                    size.text = '0.05 {}'.format(str(length))
                    material = ET.SubElement(visual, 'material')
                    script = ET.SubElement(material, 'script')
                    uri = ET.SubElement(script, 'uri')
                    uri.text = 'file://media/materials/scripts/gazebo.material'
                    name = ET.SubElement(script, 'name')
                    name.text = 'Gazebo/Yellow'
                    # ambient = ET.SubElement(material, 'ambient')
                    # ambient.text = '1 1 1 1'
                    id_line+=1


            id=0
            for station in data['Stations']:
                stationPoint=[station['Vector']['X']/1000.0, station['Vector']['Y']/1000.0]
                collision = ET.SubElement(parent, 'collision', name='station_{}'.format(id))
                pose = ET.SubElement(collision, 'pose')
                pose.text = '{} {} 0 0 0 {}'.format(str(stationPoint[0]),str(stationPoint[1]) , '0')
                geometry = ET.SubElement(collision, 'geometry')
                plane = ET.SubElement(geometry, 'plane')
                # size = ET.SubElement(plane, 'size')
                # size.text = '0.05 {}'.format(str(length))
                surface=ET.SubElement(collision,'surface')
                friction=ET.SubElement(surface,'friction')
                ode=ET.SubElement(friction,'ode')
                mu=ET.SubElement(ode,'mu')
                mu.text='100'
                mu2=ET.SubElement(ode,'mu2')
                mu2.text='50'
                torsional=ET.SubElement(friction,'torsional')
                ode2=ET.SubElement(torsional,'ode')
                contact=ET.SubElement(surface,'contact')
                ode3=ET.SubElement(contact,'ode')
                bounce=ET.SubElement(surface,'bounce')
                max_contacts=ET.SubElement(collision,'max_contacts')
                max_contacts.text='10'

                visual = ET.SubElement(parent, 'visual', name='station_{}'.format(id))
                pose = ET.SubElement(visual, 'pose')
                pose.text = '{} {} 0 0 0 {}'.format(str(stationPoint[0]),str(stationPoint[1]) , '0')
                cast_shadows=ET.SubElement(visual,'cast_shadows')
                geometry = ET.SubElement(visual, 'geometry')
                cylinder = ET.SubElement(geometry, 'cylinder')
                radius = ET.SubElement(cylinder, 'radius')
                radius.text = '0.1'
                length=ET.SubElement(cylinder,'length')
                length.text='0'
                # cylinder = ET.SubElement(geometry, 'cylinder')
                material = ET.SubElement(visual, 'material')
                script = ET.SubElement(material, 'script')
                uri = ET.SubElement(script, 'uri')
                uri.text = 'file://media/materials/scripts/gazebo.material'
                name = ET.SubElement(script, 'name')
                name.text = 'Gazebo/Blue'
                # ambient = ET.SubElement(material, 'ambient')
                # ambient.text = '1 1 1 1'

                id+=1

            self_collide=ET.SubElement(parent, 'self_collide')
            self_collide.text='0'
            kinematic=ET.SubElement(parent, 'kinematic')
            kinematic.text='0'
            gravity=ET.SubElement(parent,'gravity')
            gravity.text='1'

        except Exception as e:
            print(e)






    def generate(self):
        sdf, world = self.write_basics()

        model = ET.SubElement(world, 'model', name='obstacle')
        static = ET.SubElement(model, 'static')
        static.text = '1'
        pose = ET.SubElement(model, 'pose', frame='')
        pose.text = '0 0 0 0 0 0'
        link = ET.SubElement(model, 'link', name='obstacle')
        
        parent = link

        flag = copy(self.map) # 255 -> need to build wall, 0 -> already ...

        wall_count = 0
        while True:
            idxs = np.argwhere(flag == 255)
            if len(idxs) == 0:
                break

            wall_count += 1

            start_r, start_c = idxs[0]
            # start_r=self.height-1-start_r
            
                
            h_wall = False
            v_wall = False
            end_r = start_r
            end_c = start_c
            # if self.map[end_r + 1][end_c] == 255 :
            if self.is_v_wall_type(start_r, start_c, self.map):
                v_wall = True
                while self.map[end_r][end_c] == 255:
                    flag[end_r][end_c] = 0
                    end_r += 1
                center_x = self.origin[0] + int((start_c + end_c) / 2) * self.resolution
                center_y = self.origin[1] + int((start_r + end_r) / 2) * self.resolution
                # center_y = self.origin[1] + int(self.height-1-(start_r + end_r) / 2) * self.resolution
                l = (end_r - start_r) * self.resolution
                self.generate_v_wall(link, center_x, center_y, l, wall_count)
            elif self.map[end_r][end_c + 1] == 255:
                h_wall = True
                while self.map[end_r][end_c] == 255:
                    flag[end_r][end_c] = 0
                    end_c += 1
                center_x = self.origin[0] + int((start_c + end_c) / 2) * self.resolution
                center_y = self.origin[1] + int((start_r + end_r) / 2) * self.resolution
                l = (end_c - start_c) * self.resolution
                self.generate_h_wall(link, center_x, center_y, l, wall_count)
            else:
                flag[end_r][end_c] = 0
                continue
            print('[{} idxs remained], wall_count={}, center_x={}, center_y={}, l={}'.format(len(idxs), wall_count, center_x, center_y, l))

        ##Draw the layout in world
        model_layout = ET.SubElement(world, 'model', name='yellow_lines')
        static_layout = ET.SubElement(model_layout, 'static')
        static_layout.text = '1'
        link_layout = ET.SubElement(model_layout, 'link', name='yellow_lines')

        self.generate_layout(link_layout)



        tree = ET.ElementTree(sdf)
        # print('oi' + self.generated_world_path)

        
        tree.write(self.generated_world_path, pretty_print=True)





parser = argparse.ArgumentParser(description="Run commands")
parser.add_argument('--yaml', type=str, help="yaml path")
parser.add_argument('--layout', type=str, help="layout path")
if len(sys.argv)==1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

world_generator = WorldGenerator(args.yaml, args.layout)
world_generator.generate()
