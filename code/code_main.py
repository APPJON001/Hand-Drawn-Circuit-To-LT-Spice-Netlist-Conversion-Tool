'''
Main code that interfaces between the pipeline and the web application.
'''
import numpy as np
from PIL import Image
import yolov8 as yl
import preprocessing as pp
from processing import processing as pc
from graph import find_terminals_graph

class mainProcessing:

    def __init__(self, image):
        self.image = image
        print("Main processing object initialized")
    
    def insert_to_pipeline(self):
        try:
            
            # Save input image
            self.image.save("og_image.png", "PNG") # Save image
            print("Image saved in web_app/og_image.png") # Indicate that image has been saved
            self.og_image_path = "../web_app/og_image.png" # Original image path
            
            # Preprocess
            self.pp_image_path = "../web_app/pp_image.png" # Pre-processed image path
            pp.preProcess.global_binarize(self.og_image_path, self.pp_image_path)

            # Predict 
            self.component_detector = yl.ComponentDetection()
            self.results = self.component_detector.predict(self.pp_image_path, save = True, conf = 0.5)
            print("Results = ", self.results)
            self.xyxys, self.confidences, self.class_ids = self.component_detector.get_image_bboxes(self.results)
            
            print("xyxys = ", self.xyxys)
            
            # Generate standard prediction plot
            for r in self.results:
                im_array = r.plot()  # plot a BGR numpy array of predictions
                self.im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                self.im.save('predictions.png')  # save image            
                print("saved predictions.png")
            
            # Generate prediction plot with labelled components for web application 
            for r in self.results:
                im_array = r.plot(conf = False, labels = False)  # plot a BGR numpy array of predictions
                im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
                new_image, new_labels = pc.add_custom_labels_to_image(im, r.names, self.class_ids, self.xyxys)
                new_image.save('../web_app/website/static/images/displayed_predictions.png')  # save image  
                new_image.save('displayed_predictions.png')  # save image  
                print("saved displayed_predictions.png")
                
            return new_image, new_labels
            
        except Exception as e:
            print(f"Error processing the image: {str(e)}")
            return None, None
 

    def process_image(self, component_values, indexed_labels):
        print('Processing image!')
        print('xyxys = ', self.xyxys)    
        print('Indexed labels = ', indexed_labels)    
        print('component_values = ', component_values)    
        print('Confidences = ', self.confidences)    
        print('Class_ids = ', self.class_ids) 
        
        num_components = len(self.xyxys)

        # Get a list of intersection points between bounding boxes and the circuit schematic outline
        intersections = pc.get_imbb_intersections(self.pp_image_path, self.xyxys)
        print("intersections found")
        
        # Perform a K-Means search on the image with the bounding boxes
        terminal_locations = pc.k_means(intersections, num_components)
        print("terminal locations found")
        print("terminal locations = ", terminal_locations)

        # Remove components from the image and save the new image
        component_rem_image = pc.rem_components(self.pp_image_path, self.xyxys)
        print("removed terminal components image rendered")
        print(component_rem_image)
        
        # Erode image to make connections thicker for enhanced detection purposes
        eroded_image_path = pc.erode(component_rem_image)
        print("Image has been eroded for enhanced connection detection")
        
        # Perform edge detection on the component removed image
        edge_image_path = pc.detect_edges(eroded_image_path)
        print("edge detection complete")
        
        # Get terminal connections through pixel tracing algorithm
        #terminal_connections = pc.get_terminal_connections(edge_image_path, terminal_locations)
        print("Connections found and completed!")
        
        # Using the Graph method to determine how the terminals are connected
        terminal_connections = find_terminals_graph(eroded_image_path, terminal_locations)
        print("terminal connections found")
        print("terminal connections = ", terminal_connections)
        terminal_locations = np.array(terminal_locations, dtype = int)        
        component_labels = [label for id, name, label in indexed_labels]
        component_names = [name for id, name, label in indexed_labels]
        
        print("terminal locations = ", terminal_locations)
        print("component labels = ", component_labels)
        print("component names = ", component_names)
        print("component values = ", component_values)
        
        # Get component names mapped to their associated terminals
        component_terminals = pc.get_component_terminals(self.xyxys, component_names, terminal_locations)
        print("component_terminals = ", component_terminals)

        # Plot a graph of the detected connections
        pc.get_terminal_graph(terminal_connections, 'eroded_image.png')

        # Get node numbers mapped to their associated terminals
        node_terminal_mapping = pc.get_nodes(terminal_connections)
        print("node_terminal_mapping = ", node_terminal_mapping)
        
        # Get the component terminals in the following sequence: {component_id: [[+terminal],[-terminal]], ...}
        # This involves component orientation detection for DC and diode componpents
        pn_component_terminals = pc.find_pn_component_terminals(self.xyxys, component_terminals)
        print("\nPositive/Negative Terminal Comparison\n\ncomponent_terminals = ", component_terminals)
        print("pn_component_terminals = ", pn_component_terminals)
        
        # Get the component to positive/negative node mapping
        pn_component_node_mapping = pc.generate_component_node_mapping(pn_component_terminals, node_terminal_mapping)
        
        # Generate netlist.
        netlist_name = pc.generate_netlist(component_values, pn_component_node_mapping)
        
        # Return the netlist to be downloaded from the page!
        return netlist_name