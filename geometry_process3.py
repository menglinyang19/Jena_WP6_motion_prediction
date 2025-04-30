# This version is to fix all the lines in the geometry data along with added centerlines based on geometry_process1.py
#   1. for nodes belonging to different segments, recording them seperately as different nodes
#   2. when create Lanecenters, their directions should be consistent with the right-side way
#   3. creating lanecenters for all lanelets(check first)
#   4. iterate through regulatory_element and multipolygon first, lanelet then, to encode ways. 
#   5. re-order the sequence of recording nodes
import numpy as np
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict
import os
import pandas as pd
import utm
import csv
from collections import defaultdict
from shapely.geometry import LineString
import random

def lat_lon_to_local(lat, lon, xUtmOrigin, yUtmOrigin):
    # Convert latitude and longitude strings to floating-point numbers
    lat_float = float(lat)
    lon_float = float(lon)
    # Convert input latitude and longitude to UTM coordinates
    utm_coords = utm.from_latlon(lat_float, lon_float)
    easting, northing = utm_coords[0], utm_coords[1]
    
    # Calculate local coordinates by subtracting the origin UTM coordinates
    local_easting = easting - xUtmOrigin
    local_northing = northing - yUtmOrigin
    
    return local_easting, local_northing

def get_location_info(agent_data_dictionary):
    location_id_collect = []
    location_collection = []
    for i in range(33):
        recording_filename = f"{i:02d}_recordingMeta.csv"
        recording_filepath = os.path.join(agent_data_dictionary, recording_filename)
        df = pd.read_csv(recording_filepath)
        recording_data = df.values

        location_id = recording_data[0,1]

        if location_id not in location_id_collect:
            location_coor = recording_data[0,12:14]
            location_id_collect.append(location_id)
            location_collection.append([location_id, *location_coor])

    location_collection = np.array(location_collection)
    sorted_indices = np.argsort(location_collection[:, 0])
    location_collection_sorted = location_collection[sorted_indices]
    return location_collection_sorted

# Define a function to get node IDs from a way
def get_node_ids_from_way(way):
    node_ids = []
    for nd in way.findall('nd'):
        node_ids.append(nd.get('ref'))
    return node_ids

def calculate_centerline_points(polyline1, polyline2, subtype, one_way_indicator):
    # Calculate the direction vectors for each polyline
    # lanecenter will take the direction of polyline1
    vector1 = np.array(polyline1[-1]) - np.array(polyline1[0])
    vector2 = np.array(polyline2[-1]) - np.array(polyline2[0])

    if np.any(vector1) and np.any(vector2):
        # Calculate the angle between the direction vectors
        denominator = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        angle = np.arccos(np.dot(vector1, vector2) / (denominator + 1e-6))

        # Determine the direction of the centerline based on the angle
        if -np.pi / 2 <= angle <= np.pi / 2:
            centerline_direction = "Clockwise"
        else:
            centerline_direction = "Counter-clockwise"

        # Reverse the direction of polyline2 if necessary
        if centerline_direction == "Counter-clockwise":
            polyline2 = list(reversed(polyline2))

        # Determine the number of points in each polyline
        num_points1 = len(polyline1)
        num_points2 = len(polyline2)
        
        # Determine the maximum number of points
        max_num_points = max(num_points1, num_points2)
        
        # Interpolate points along the shorter polyline to match the length of the longer one
        if num_points1 < num_points2:
            polyline1 = interpolate_points(polyline1, max_num_points)
        elif num_points2 < num_points1:
            polyline2 = interpolate_points(polyline2, max_num_points)

        # Create a list to store the centerline points
        centerline_points = []

        # Iterate through the points on the polylines and find the midpoints
        for p1, p2 in zip(polyline1, polyline2):
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            centerline_points.append(midpoint)

        # Create the centerline polyline
        centerline = np.array(centerline_points)

        # for the bi-directional way
        if subtype != "walkway" and one_way_indicator == "no":
            reversed_centerline = list(reversed(centerline))

            #####
                #check which dimension should two lines concatenate
            #####
            centerline = np.dstack((centerline, reversed_centerline))
        else:
            centerline = centerline[:, :, np.newaxis]

        return centerline
    else:
        return None

def interpolate_points(polyline, target_length):
    """
    Interpolates points along the polyline to match the target length.
    """
    num_points = len(polyline)
    interpolated_polyline = []
    
    # Calculate the interpolation factor
    interpolation_factor = (num_points - 1) / (target_length - 1)
    
    # Interpolate points along the polyline
    for i in range(target_length):
        index = int(i * interpolation_factor)
        interpolated_polyline.append(polyline[index])
    
    return interpolated_polyline

def get_geometry_data(agent_data_dictionary,location_dictionary):

    relationtype_file = '/home/meya174e/bin/inD_data/lanelets/relation_records.xlsx'
    lanelet_type = pd.read_excel(relationtype_file, sheet_name='lanelet', header=None)
    lanelet_type = lanelet_type.to_numpy()[1:,:]
    multipoly_type = pd.read_excel(relationtype_file, sheet_name='multipolygon', header=None)
    multipoly_type = multipoly_type.to_numpy()[1:,:]
    RE_type = pd.read_excel(relationtype_file, sheet_name='regulatory_element', header=None)
    RE_type = RE_type.to_numpy()[1:,:]

    # get locations' geometry data
    node_info = {}
    lanecenter = {}
    start_id = 0
    num_digits = 7

    for j in range(4):
        location_filename = f"location{j+1}.osm"
        location_filepath = os.path.join(location_dictionary, location_filename)
        location_collection = get_location_info(agent_data_dictionary)

        # get the coordinates of UTM origin
        origin_coor = location_collection[j,1:3]
        # Load the XML file
        tree = ET.parse(location_filepath)
        root = tree.getroot()
        
        """
           get basic node information from original geometry data
        """ 
        # extract GPS of all points and transfer them to UTM and then local coordinates
        node_info_j = {}
        node_id = []
        node_xyz = []
        for node in root.findall('node'):
            node_id.append(node.get('id'))
            lat = node.get('lat')
            lon = node.get('lon')
            x, y = lat_lon_to_local(lat, lon, origin_coor[0], origin_coor[1])
            node_xyz.append([x,y,-1])
        node_id = np.array(node_id).reshape(-1,1)
        node_xyz = np.array(node_xyz)
        
        num_node = len(node_id)
        # node_type records the type of ways (encoded) where nodes are located
        node_type = -1 * np.ones(num_node)
        # node_id records the id of ways (encoded) where nodes are located
        vector_id = [np.array([], dtype=object) for _ in range(num_node)]

        """
           go through all relations to
           1. get way_type (encoding) from all relations
           2. create lanecenters for all lanelets
               1). create nodes and ways of lanecenters
               2). encoding and recording all these new nodes
        """ 
        # initiate to get way_type
        way_id = []
        for way in root.findall('way'):
            # Extract way ID
            way_id.append(way.get('id'))
        way_id = np.array(way_id).reshape(-1,1)
        num_way = way_id.shape[0]
        way_type = -1 * np.ones(num_way)

        # initiate the records of new nodes created to form lane centers
        start_id_node = 0
        new_node_id = []
        new_node_xyz = []
        new_node_type = []
        new_vector_id = []
     
        lanecenter = {}
        # iterate through all relations
        for relation in root.findall('relation'): 
            relation_id = relation.get('id')
            # Check if the relation has the specified tag
            type_tag = relation.find("./tag[@k='type']")
            type = type_tag.get('v') if type_tag is not None else None
            if type == "lanelet":
                # Check if there is a subtype tag with the desired values
                subtype_tag = relation.find("./tag[@k='subtype']")
                subtype = subtype_tag.get('v') if subtype_tag is not None else None

                one_way_tag = relation.find("./tag[@k='one_way']")
                one_way_indicator = one_way_tag.get('v') if one_way_tag is not None else None

                encoding_index = np.where((lanelet_type[:,0] == subtype) & (lanelet_type[:,2] == one_way_indicator))[0][0]

                bounds = []
                bounds_location = []
                
                # encode the composed ways first
                # Extract the two boundaries of the lanelet from the relation's members
                members = relation.findall('member')
                for member in members:
                    if member.get('type') == 'way':
                        bounds.append(member.get('ref'))
                        bounds_location.append(member.get('role')) 
                
                if len(bounds) == 2:
                    if bounds_location[0] != bounds_location[1]:
                    # the lanecenter will take the direction of the right bound
                        if bounds_location[0] == "right":
                            way_0_id = bounds[0]
                            way_0 = root.find(f".//way[@id='{way_0_id}']")
                            way_1_id = bounds[1] 
                            way_1 = root.find(f".//way[@id='{way_1_id}']")
                        else:   
                            way_0_id = bounds[1]
                            way_0 = root.find(f".//way[@id='{way_0_id}']")
                            way_1_id = bounds[0]
                            way_1 = root.find(f".//way[@id='{way_1_id}']")

                        if way_0 is not None:

                            # fill the way type
                            index0 = np.where(way_id == way_0_id)[0][0]
                            way_type[index0] = max(lanelet_type[encoding_index,5], way_type[index0]) # check relation_record.xlsx

                            # get all the nodes on the bound
                            node_id_polyline_0 = get_node_ids_from_way(way_0)
                            polyline0= []
                            for node_id_polyline_0_item in node_id_polyline_0:
                                try:
                                    index = np.where(node_id == node_id_polyline_0_item)[0][0]
                                    # Find the coordinates of the node using its index
                                    coordinates = node_xyz[index]
                                    polyline0.append(coordinates)
                                except ValueError:
                                    print("Node ID", node_id_polyline_0_item, "not found in node_id array.")
                            polyline0 = np.array(polyline0)[:,:2]

                        if way_1 is not None:

                            # fill the way type
                            index1 = np.where(way_id == way_1_id)[0][0]
                            way_type[index1] = max(lanelet_type[encoding_index,5],way_type[index1]) # check relation_record.xlsx

                            # get all the nodes on the bound
                            node_id_polyline_1 = get_node_ids_from_way(way_1)
                            polyline1= []
                            for node_id_polyline_1_item in node_id_polyline_1:
                                try:
                                    index = np.where(node_id == node_id_polyline_1_item)[0][0]
                                    # Find the coordinates of the node using its index
                                    coordinates = node_xyz[index]
                                    polyline1.append(coordinates)
                                except ValueError:
                                    print("Node ID", node_id_polyline_1_item, "not found in node_id array.")
                            polyline1 = np.array(polyline1)[:,:2]
                    
                        centerline = calculate_centerline_points(polyline0, polyline1, subtype, one_way_indicator)

                        if centerline is not None:   
                            for k in range(centerline.shape[2]):
                                new_way_id = "-" + str(start_id).zfill(num_digits)
                                start_id += 1
                                lanecenter[start_id] = {"way_id": new_way_id,
                                    "bounds": bounds,
                                    "centerline": centerline[:,:,k],
                                    "bounds_type": subtype,
                                    "way_type": lanelet_type[encoding_index,7]
                                }

                                num1_node = centerline[:,:,k].shape[0]
                                for m in range(num1_node):
                                    new_id_node = "-" + str(start_id_node).zfill(num_digits)
                                    new_node_id.append(new_id_node)
                                    new_node_xyz.append([centerline[m,0,k], centerline[m,1,k], -1]) 
                                    start_id_node += 1
                                    new_node_type.append(lanelet_type[encoding_index,7])
                                    new_vector_id.append(new_way_id)

            elif type == "multipolygon":
                # Check if there is a subtype tag with the desired values
                subtype_tag = relation.find("./tag[@k='subtype']")
                subtype = subtype_tag.get('v') if subtype_tag is not None else None 
                encoding_index = np.where(multipoly_type[:,0] == subtype)[0][0]

                members = relation.findall('member')
                for member in members:
                    if member.get('type') == 'way':
                        index = np.where(way_id == member.get('ref'))[0][0]
                        way_type[index] = max(multipoly_type[encoding_index,3], way_type[index]) 
                    elif member.get('type') == 'way':
                        index_node = np.where(node_id == member.get('ref'))[0][0]
                        node_type[index_node] = max(multipoly_type[encoding_index,3], way_type[index])

            elif type == "regulatory_element":
                members = relation.findall('member')
                for member in members:
                    member_role =  member.get('role')
                    encoding_index = np.where(RE_type[:,1] == member_role)[0][0]
                    if member.get('type') == 'way':
                        index = np.where(way_id == member.get('ref'))[0][0]
                        way_type[index] = max(RE_type[encoding_index,2], way_type[index]) 
                    elif member.get('type') == 'relation':
                        ref_id = member.get('ref')
                        nested_relation = root.find(f".//relation[@id='{ref_id}']")
                        nested_members = nested_relation.findall('member')
                        for nested_member in nested_members:
                            if nested_member.get('type') == 'way':
                                index = np.where(way_id == nested_member.get('ref'))[0][0]
                                way_type[index] = max(RE_type[encoding_index,2], way_type[index]) 
                
        new_node_id = np.array(new_node_id)
        new_node_xyz = np.array(new_node_xyz)
        new_node_type = np.array(new_node_type)
        new_vector_id = np.array(new_vector_id)

        unique_node_types = np.unique(new_node_type)

        # go through all ways to encoding and sorting all original nodes
        sorted_node_id = []
        sorted_node_xyz = []
        sorted_node_type = []
        sorted_vector_id = []
        num_way = way_id.shape[0]
        for i in range(num_way):
            ref_id = way_id[i,0]
            ref_type = way_type[i]
            way = root.find(f".//way[@id='{ref_id}']")
            # Iterate over each point (nd element) in the way and save them in this sequence
            for nd in way.findall('nd'):
                node_id_i = nd.get('ref')  
                # Find the index of node_id_i in node_id
                index_node = np.where(node_id == node_id_i)[0][0]
                
                sorted_node_id.append(node_id_i)
                sorted_node_xyz.append(node_xyz[index_node])
                sorted_node_type.append(ref_type)
                sorted_vector_id.append(ref_id)
            
        sorted_node_id = np.array(sorted_node_id)   
        sorted_node_xyz = np.array(sorted_node_xyz)    
        sorted_node_type = np.array(sorted_node_type)
        sorted_vector_id = np.array(sorted_vector_id)  

        empty_indices = []
        # Iterate over each element of vector_id and find the empty ones
        for i, element in enumerate(vector_id):
            if not element:  # Check if the element is empty (i.e., an empty list)
                vector_id[i] = "-1"  # Set the empty element to the string "-1"

                # Optionally, you can store the index of the modified element
                empty_indices.append(i)
        vector_id = np.array(vector_id).reshape(-1,1)

        # combine the original nodes and new created nodes
        combined_node_id = np.concatenate((sorted_node_id, new_node_id), axis=0)
        combined_node_xyz = np.concatenate((sorted_node_xyz, new_node_xyz), axis=0)
        combined_vector_id = np.concatenate((sorted_vector_id, new_vector_id), axis=0)
        combined_node_type = np.concatenate((sorted_node_type, new_node_type.squeeze()), axis=0)
        combined_node_type = combined_node_type.astype(int)

        num_combined_node = combined_node_id.shape[0]
        combined_node_valid = np.zeros((num_combined_node,1))
        # Ensure all arrays have the same number of rows using assert
        assert combined_node_xyz.shape[0] == combined_vector_id.shape[0] == combined_node_type.shape[0] == combined_node_valid.shape[0], (
            f"Shape mismatch: combined_node_xyz: {combined_node_xyz.shape}, "
            f"combined_vector_id: {combined_vector_id.shape}, "
            f"combined_node_type: {combined_node_type.shape}, "
            f"combined_node_valid: {combined_node_valid.shape}"
        )
        combined_node_valid[combined_node_type > 0] = 1  # Set to 0 where combined_node_type is valid (not -1)
        # Iterate over the indices of empty elements in vector_id
        for i in empty_indices:
            combined_node_valid[i] = 0  # Set the corresponding position in combined_node_valid to 0

        node_info_j = {
            "roadgraph_samples_node/id": combined_node_id,
            "roadgraph_samples/id": combined_vector_id,
            "roadgraph_samples/xyz": combined_node_xyz,
            "roadgraph_samples/type":combined_node_type,
            "roadgraph_samples/valid": combined_node_valid
        }
        
        key_location = f"Location_{j+1}"
        node_info[key_location] = node_info_j

    location_dictionary = '/home/meya174e/bin/inD_data/lanelets/'
    geometry_filename = "geometry_info6.npz"
    geometry_filepath = os.path.join(location_dictionary, geometry_filename)
    np.savez(geometry_filepath, **node_info)

    return node_info
