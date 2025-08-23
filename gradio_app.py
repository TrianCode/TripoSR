# import logging
# import os
# import tempfile
# import time

# import gradio as gr
# import numpy as np
# import rembg
# import torch
# import trimesh
# from PIL import Image
# from functools import partial
# import plotly.graph_objects as go
# import plotly.express as px
# import json

# from tsr.system import TSR
# from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
# from tsr.evaluation import calculate_metrics

# import argparse


# # Configure CUDA memory settings
# if torch.cuda.is_available():
#     device = "cuda:0"
#     # Lower default chunk size to reduce memory usage
#     default_chunk_size = 8192
# else:
#     device = "cpu"
#     default_chunk_size = 8192

# # model = TSR.from_pretrained(
# #     "stabilityai/TripoSR",
# #     config_name="config.yaml",
# #     weight_name="model.ckpt",
# # )

# print("Loading custom TripoSR model...")
# model_custom = TSR.from_pretrained(
#     "TrianC0de/TripoSR2",
#     config_name="config.yaml",
#     weight_name="sf3d_checkpoint_epoch_3000.ckpt", 
# )

# # adjust the chunk size to balance between speed and memory usage
# model.renderer.set_chunk_size(default_chunk_size)
# model.to(device)

# rembg_session = rembg.new_session()

# # Global storage for historical metrics to enable comparison
# metrics_history = []

# def create_metrics_radar_chart(current_metrics):
#     """Create a radar chart comparing the current metrics with historical averages"""
#     # Define metrics to show (lower is better for UHD, TMD, CD; higher is better for IoU and F1)
#     metrics_to_show = {
#         'f1_score': {'display': 'F1', 'invert': False},
#         'uniform_hausdorff_distance': {'display': 'UHD', 'invert': True},
#         'tangent_space_mean_distance': {'display': 'TMD', 'invert': True},
#         'chamfer_distance': {'display': 'CD', 'invert': True},
#         'iou_score': {'display': 'IoU', 'invert': False}
#     }
    
#     # Filter metrics_to_show to only include keys that exist in current_metrics
#     available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
#     # If we have no available metrics or no historical metrics, return empty chart
#     if not available_metrics or len(metrics_history) == 0:
#         # Create an empty figure with a message if no history
#         fig = go.Figure()
#         fig.add_annotation(
#             text="Generate more models to see comparison with historical average",
#             xref="paper", yref="paper",
#             x=0.5, y=0.5,
#             showarrow=False
#         )
#         fig.update_layout(title="Metrics Comparison")
#         return fig
    
#     # Calculate average of historical metrics
#     avg_metrics = {}
#     for metric_name in available_metrics.keys():
#         # Check if all historical metrics have this key
#         valid_hist = [hist for hist in metrics_history if metric_name in hist]
#         if valid_hist:
#             avg_metrics[metric_name] = sum(hist[metric_name] for hist in valid_hist) / len(valid_hist)
#         else:
#             # If no historical data has this metric, use current value
#             avg_metrics[metric_name] = current_metrics[metric_name]
    
#     # Create data for the radar chart
#     categories = [available_metrics[m]['display'] for m in available_metrics.keys()]
    
#     # Normalize values for better visualization (invert where necessary)
#     current_values = []
#     history_values = []
    
#     for metric_name, config in available_metrics.items():
#         # Get raw values
#         current_val = current_metrics[metric_name]
#         avg_val = avg_metrics[metric_name]
        
#         # For metrics where lower is better, invert for visualization
#         if config['invert']:
#             # Use a simple inversion formula for normalized values
#             # Map to 0-1 scale where 1 is better
#             max_val = max(current_val, avg_val) * 1.2  # 20% buffer
#             current_values.append(1 - (current_val / max_val))
#             history_values.append(1 - (avg_val / max_val))
#         else:
#             current_values.append(current_val)
#             history_values.append(avg_val)
    
#     # Create the radar chart
#     fig = go.Figure()
    
#     # Add current metrics
#     fig.add_trace(go.Scatterpolar(
#         r=current_values,
#         theta=categories,
#         fill='toself',
#         name='Current Model'
#     ))
    
#     # Add historical average
#     fig.add_trace(go.Scatterpolar(
#         r=history_values,
#         theta=categories,
#         fill='toself',
#         name='Historical Average'
#     ))
    
#     # Update layout
#     fig.update_layout(
#         polar=dict(
#             radialaxis=dict(
#                 visible=True,
#                 range=[0, 1]
#             )
#         ),
#         showlegend=True,
#         title="Metrics Comparison (Higher is Better)"
#     )
    
#     return fig

# def create_metrics_bar_chart(current_metrics):
#     """Create a bar chart for current metrics"""
#     metrics_to_show = {
#         'f1_score': {'display': 'F1 Score (↑)', 'color': 'purple'},
#         'uniform_hausdorff_distance': {'display': 'UHD (↓)', 'color': 'red'},
#         'tangent_space_mean_distance': {'display': 'TMD (↓)', 'color': 'orange'},
#         'chamfer_distance': {'display': 'CD (↓)', 'color': 'green'},
#         'iou_score': {'display': 'IoU (↑)', 'color': 'blue'}
#     }
    
#     # Filter to only include metrics that exist in current_metrics
#     available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
#     if not available_metrics:
#         # Create an empty figure with a message if no metrics available
#         fig = go.Figure()
#         fig.add_annotation(
#             text="No metrics available",
#             xref="paper", yref="paper",
#             x=0.5, y=0.5,
#             showarrow=False
#         )
#         fig.update_layout(title="Metrics")
#         return fig
    
#     # Create lists for the bar chart
#     names = [available_metrics[m]['display'] for m in available_metrics.keys()]
#     values = [current_metrics[m] for m in available_metrics.keys()]
#     colors = [available_metrics[m]['color'] for m in available_metrics.keys()]
    
#     # Create the bar chart
#     fig = go.Figure(data=[
#         go.Bar(
#             x=names,
#             y=values,
#             marker_color=colors
#         )
#     ])
    
#     # Update layout
#     fig.update_layout(
#         title="Current Metrics",
#         xaxis_title="Metric",
#         yaxis_title="Value",
#         yaxis=dict(
#             title="Value",
#             titlefont_size=16,
#             tickfont_size=14,
#         )
#     )
    
#     return fig

# def check_input_image(input_image):
#     if input_image is None:
#         raise gr.Error("No image uploaded!")


# def preprocess(input_image, do_remove_background, foreground_ratio):
#     def fill_background(image):
#         image = np.array(image).astype(np.float32) / 255.0
#         image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
#         return Image.fromarray((image * 255.0).astype(np.uint8))

#     if do_remove_background:
#         image = input_image.convert("RGB")
#         image = remove_background(image, rembg_session)
#         image = resize_foreground(image, foreground_ratio)
#         image = fill_background(image)
#     else:
#         image = input_image
#         if image.mode == "RGBA":
#             image = fill_background(image)
            
#     # Ensure image size is reasonable
#     max_size = 512
#     if max(image.size) > max_size:
#         ratio = max_size / max(image.size)
#         new_size = tuple(int(dim * ratio) for dim in image.size)
#         image = image.resize(new_size, Image.Resampling.LANCZOS)
        
#     return image


# def fix_model_orientation(mesh):
#     """Fix the orientation of the model for proper display"""
#     # Rotate 90 degrees around X axis to match standard orientation
#     rotation_matrix = trimesh.transformations.rotation_matrix(
#         angle=np.pi/2,
#         direction=[1, 0, 0],
#         point=[0, 0, 0]
#     )
#     mesh.apply_transform(rotation_matrix)
    
#     # Center the mesh
#     mesh.vertices -= mesh.vertices.mean(axis=0)
    
#     # Scale to fit in a unit cube
#     scale = 1.0 / max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
#     mesh.vertices *= scale
    
#     # Fix normals to ensure proper rendering
#     mesh.fix_normals()
    
#     # Ensure material properties aren't too reflective
#     if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
#         # Reduce specularity to minimize bright reflections
#         if hasattr(mesh.visual.material, 'specular'):
#             mesh.visual.material.specular = [0.1, 0.1, 0.1, 1.0]
#         # Set ambient color to ensure better visibility
#         if hasattr(mesh.visual.material, 'ambient'):
#             mesh.visual.material.ambient = [0.6, 0.6, 0.6, 1.0]
#         # Adjust shininess to reduce glossy appearance
#         if hasattr(mesh.visual.material, 'shininess'):
#             mesh.visual.material.shininess = 0.1

#     return mesh


# def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb"], 
#              model_quality="Standar", texture_quality=7, smoothing_factor=0.3):
#     try:
#         # Clear CUDA cache before starting
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         # Create a permanent output directory
#         output_dir = os.path.join(os.getcwd(), "outputs")
#         os.makedirs(output_dir, exist_ok=True)
        
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
        
#         # Fix: Changed quality settings keys to match the model_quality parameter default value
#         quality_settings = {
#             "Konsep": {"chunk_size": 32768, "detail_factor": 0.5},
#             "Standar": {"chunk_size": 16384, "detail_factor": 0.7},  # Changed from "Standard" to "Standar"
#             "Tinggi": {"chunk_size": 8192, "detail_factor": 1.0}
#         }
        
#         model.renderer.set_chunk_size(quality_settings[model_quality]["chunk_size"])
        
#         with torch.inference_mode():
#             scene_codes = model(image, device=device)
#             mesh = model.extract_mesh(
#                 scene_codes, 
#                 True, 
#                 resolution=min(mc_resolution, 192)
#             )[0]
        
#         mesh = to_gradio_3d_orientation(mesh)
#         mesh = fix_model_orientation(mesh)
        
#         # Apply smoothing if requested - using the proper method
#         if smoothing_factor > 0:
#             # Using laplacian_smooth instead of smoothed with proper parameters
#             # This is the correct way to apply smoothing in trimesh
#             if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
#                 from trimesh import smoothing
#                 # Apply laplacian smoothing with the specified factor as iterations
#                 iterations = max(1, int(smoothing_factor * 10))  # Convert factor to iterations (1-10)
#                 smoothing.filter_laplacian(mesh, iterations=iterations)
        
#         # Improve texture appearance by normalizing colors
#         if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
#             # Get vertex colors
#             colors = mesh.visual.vertex_colors
            
#             # Normalize brightness to prevent extreme bright spots
#             # Convert to HSV for better manipulation
#             import colorsys
#             normalized_colors = np.zeros_like(colors)
            
#             for i in range(len(colors)):
#                 r, g, b = colors[i][0]/255.0, colors[i][1]/255.0, colors[i][2]/255.0
#                 h, s, v = colorsys.rgb_to_hsv(r, g, b)
                
#                 # Cap brightness (v) to prevent overly bright spots
#                 v = min(v, 0.95)
                
#                 # Increase saturation slightly for better visual appeal
#                 s = min(s * 1.1, 1.0)
                
#                 r, g, b = colorsys.hsv_to_rgb(h, s, v)
#                 normalized_colors[i][0] = int(r * 255)
#                 normalized_colors[i][1] = int(g * 255)
#                 normalized_colors[i][2] = int(b * 255)
#                 normalized_colors[i][3] = colors[i][3]  # Keep alpha channel
            
#             mesh.visual.vertex_colors = normalized_colors
        
#         # Load reference model if provided
#         reference_mesh = None
#         if reference_model is not None:
#             reference_mesh = trimesh.load(reference_model.name)
        
#         # Calculate actual metrics
#         metrics = calculate_metrics(mesh, reference_mesh)
        
#         # Add current metrics to history (limit to last 10)
#         global metrics_history
#         metrics_history.append(metrics)
#         if len(metrics_history) > 10:
#             metrics_history = metrics_history[-10:]
        
#         # Create visualization figures
#         radar_chart = create_metrics_radar_chart(metrics)
#         bar_chart = create_metrics_bar_chart(metrics)
        
#         # Format metrics text
#         if reference_mesh is not None:
#             metrics_text = f"Metrics (compared to reference model):\n"
#             if 'f1_score' in metrics:
#                 metrics_text += f"F1 Score: {metrics['f1_score']:.4f}\n"
#             if 'uniform_hausdorff_distance' in metrics:
#                 metrics_text += f"Uniform Hausdorff Distance: {metrics['uniform_hausdorff_distance']:.4f}\n"
#             if 'tangent_space_mean_distance' in metrics:
#                 metrics_text += f"Tangent-Space Mean Distance: {metrics['tangent_space_mean_distance']:.4f}\n"
#             if 'chamfer_distance' in metrics:
#                 metrics_text += f"Chamfer Distance: {metrics['chamfer_distance']:.4f}\n"
#             if 'iou_score' in metrics:
#                 metrics_text += f"IoU Score: {metrics['iou_score']:.4f}"
#             elif 'iou' in metrics:  # Check for iou key as a fallback
#                 metrics_text += f"IoU Score: {metrics['iou']:.4f}"
#         else:
#             metrics_text = f"Self-evaluation metrics:\n"
#             if 'f1_score' in metrics:
#                 metrics_text += f"F1 Score: {metrics['f1_score']:.4f}\n"
#             if 'uniform_hausdorff_distance' in metrics:
#                 metrics_text += f"Uniform Hausdorff Distance: {metrics['uniform_hausdorff_distance']:.4f}\n"
#             if 'tangent_space_mean_distance' in metrics:
#                 metrics_text += f"Tangent-Space Mean Distance: {metrics['tangent_space_mean_distance']:.4f}\n"
#             if 'chamfer_distance' in metrics:
#                 metrics_text += f"Chamfer Distance: {metrics['chamfer_distance']:.4f}\n"
#             if 'iou_score' in metrics:
#                 metrics_text += f"IoU Score: {metrics['iou_score']:.4f}\n"
#             elif 'iou' in metrics:  # Check for iou key as a fallback
#                 metrics_text += f"IoU Score: {metrics['iou']:.4f}\n"
#             metrics_text += f"Note: For more accurate metrics, provide a reference model."
        
#         # Save files with permanent paths
#         rv = []
#         for format in formats:
#             file_path = os.path.join(output_dir, f"model_{timestamp}.{format}")
#             if format == "glb":
#                 mesh.export(file_path, file_type="glb")
#             else:
#                 # For OBJ, use improved texture settings
#                 mesh.export(
#                     file_path,
#                     file_type="obj",
#                     include_texture=True,
#                     include_normals=True,  # Ensure normals are included for better rendering
#                     resolver=None,
#                     mtl_name=f"model_{timestamp}.mtl"
#                 )
#             rv.append(file_path)
        
#         # Add metrics to return values
#         rv.extend([
#             metrics.get("f1_score", 0.0),
#             metrics.get("uniform_hausdorff_distance", 0.0),
#             metrics.get("tangent_space_mean_distance", 0.0),
#             metrics.get("chamfer_distance", 0.0),
#             # Try iou_score first, then fall back to iou if needed
#             metrics.get("iou_score", metrics.get("iou", 0.0)),
#             metrics_text,
#             radar_chart,
#             bar_chart
#         ])
        
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         return rv
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             raise gr.Error("GPU memory error. Try 'Konsep' quality or lower resolution.")
#         else:
#             raise gr.Error(f"Generation error: {str(e)}")
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")


# def run_example(image_pil):
#     preprocessed = preprocess(image_pil, False, 0.9)
#     mesh_obj, mesh_glb, f1, uhd, tmd, cd, iou, metrics_text, radar_chart, bar_chart = generate(
#         preprocessed, 128, None, ["obj", "glb"],
#         "Standar", 7, 0.3
#     )
#     return preprocessed, mesh_obj, mesh_glb, f1, uhd, tmd, cd, iou, metrics_text, radar_chart, bar_chart


# with gr.Blocks(title="3D Model Generation") as interface:
#     gr.Markdown(
#         """    
# # Generasi Model 3D dari Gambar

# Unggah gambar untuk menghasilkan model 3D dengan parameter yang dapat disesuaikan.

# ## Fine-Tuning Parameters

# - **Foreground Ratio**: Mengontrol seberapa banyak gambar yang dianggap sebagai latar depan saat pemrosesan. Nilai lebih tinggi akan lebih fokus pada objek utama.
# - **Marching Cubes Resolution**: Mengontrol tingkat detail mesh 3D. Nilai lebih tinggi menciptakan model lebih detail tetapi membutuhkan daya pemrosesan lebih besar.
# - **Kualitas Model**: Mengatur tingkat kualitas keseluruhan, mempengaruhi waktu pemrosesan dan detail hasil:
#   - Draft: Lebih cepat tapi kurang detail
#   - Standar: Pilihan seimbang untuk kebanyakan kasus
#   - Tinggi: Lebih detail tapi pemrosesan lebih lambat
# - **Kualitas Tekstur**: Mengontrol detail tekstur yang diterapkan pada model. Nilai lebih tinggi menciptakan tekstur lebih detail.
# - **Mesh Smoothing**: Menerapkan penghalusan pada model akhir. Nilai lebih tinggi menciptakan permukaan lebih halus tapi mungkin kehilangan detail halus.

# ## Tips:
# 1. Jika hasil tidak memuaskan, coba sesuaikan parameter Foreground Ratio dan Mesh Smoothing.
# 2. Untuk model lebih detail, tingkatkan Marching Cubes Resolution dan atur Kualitas Model ke "Tinggi".
# 3. Lebih baik nonaktifkan "Hapus Latar Belakang" untuk contoh yang disediakan (kecuali yang terakhir) karena sudah diproses sebelumnya.
# 4. Nonaktifkan opsi "Hapus Latar Belakang" hanya jika gambar input Anda adalah RGBA dengan latar belakang transparan, konten gambar terpusat dan menempati lebih dari 70% lebar atau tinggi gambar.
# 5. Untuk metrik evaluasi yang akurat, unggah model referensi dalam format OBJ, GLB atau STL.
# 6. Waktu pemrosesan meningkat dengan pengaturan resolusi dan kualitas yang lebih tinggi.
#     """
#     )
#     with gr.Row(variant="panel"):
#         with gr.Column():
#             with gr.Row():
#                 input_image = gr.Image(
#                     label="Gambar Input",
#                     image_mode="RGBA",
#                     sources="upload",
#                     type="pil",
#                     elem_id="content_image",
#                 )
#                 processed_image = gr.Image(label="Processed Image", interactive=False)
#             with gr.Row():
#                 with gr.Group():
#                     do_remove_background = gr.Checkbox(
#                         label="Hapus Latar Belakang", value=True
#                     )
#                     foreground_ratio = gr.Slider(
#                         minimum=0.5,
#                         maximum=1.0,
#                         value=0.9,
#                         step=0.05,
#                         label="Foreground Ratio",
#                     )
#                     mc_resolution = gr.Slider(
#                         minimum=64,
#                         maximum=256,
#                         value=128,
#                         step=16,
#                         label="Resolusi Marching Cubes",
#                     )
#                     model_quality = gr.Dropdown(
#                         ["Konsep", "Standar", "Tinggi"],  # Changed from ["Draft", "Standard", "High"]
#                         value="Standar",  # Changed from "Standard"
#                         label="Model Quality",
#                     )
#                     texture_quality = gr.Slider(
#                         minimum=1,
#                         maximum=10,
#                         value=7,
#                         step=1,
#                         label="Kualitas Tekstur",
#                     )
#                     smoothing_factor = gr.Slider(
#                         minimum=0.0,
#                         maximum=1.0,
#                         value=0.3,
#                         step=0.1,
#                         label="Mesh Smoothing",
#                     )
#                     reference_model = gr.File(
#                         label="Reference Model (OBJ/GLB/STL) [optional]", 
#                         file_types=[".obj", ".glb", ".stl"]
#                     )
#                     submit = gr.Button("Generate 3D Model", variant="primary")
#                     evaluation_info = gr.Button("ℹ️ Metric Information", size="sm")
        
#         with gr.Column():
#             with gr.Tabs():
#                 with gr.TabItem("3D Visualization"):
#                     output_model_obj = gr.Model3D(
#                         label="Model 3D (OBJ)",
#                         interactive=False
#                     )
#                     output_model_glb = gr.Model3D(
#                         label="Model 3D (GLB)",
#                         interactive=False
#                     )
#                 with gr.TabItem("Metrik Evaluasi"):
#                     with gr.Row():
#                         f1_metric = gr.Number(label="F1 Score", value=0.0, precision=4)
#                         uhd_metric = gr.Number(label="Uniform Hausdorff Distance", value=0.0, precision=4)
#                         tmd_metric = gr.Number(label="Tangent-Space Mean Distance", value=0.0, precision=4)
#                         cd_metric = gr.Number(label="Chamfer Distance", value=0.0, precision=4)
#                         iou_metric = gr.Number(label="IoU Score", value=0.0, precision=4)
                    
#                     with gr.Row():
#                         metrics_text = gr.Textbox(
#                             label="Metrik Lengkap", 
#                             value="Hasilkan model untuk melihat metrik evaluasi.\n\nUntuk perbandingan yang lebih akurat, unggah model referensi.",
#                             lines=6
#                         )
                
#                 with gr.TabItem("Visualisasi Metrik"):
#                     gr.Markdown("""
#                     ### Visualisasi Perbandingan Metrik
                    
#                     Diagram di bawah menunjukkan perbandingan metrik model saat ini dengan rata-rata historis.
#                     Semakin besar nilai pada diagram radar, semakin baik kualitas metrik tersebut.
#                     """)
#                     with gr.Row():
#                         radar_plot = gr.Plot(label="Perbandingan dengan Riwayat", show_label=False)
                    
#                     gr.Markdown("""
#                     ### Nilai Metrik Saat Ini
                    
#                     Diagram batang di bawah menunjukkan nilai absolut dari metrik saat ini.
#                     UHD, TMD, CD: nilai lebih rendah lebih baik (↓)
#                     IoU: nilai lebih tinggi lebih baik (↑)
#                     """)
#                     with gr.Row():
#                         bar_plot = gr.Plot(label="Nilai Metrik Saat Ini", show_label=False)
                    
#                     gr.Markdown("""
#                     **Petunjuk Metrik:**
#                     - **F1 Score**: Mengukur keseimbangan antara presisi dan recall. Nilai lebih tinggi (0-1) menunjukkan kecocokan permukaan yang lebih baik.
#                     - **Uniform Hausdorff Distance (UHD)**: Mengukur jarak maksimum antara permukaan mesh. Nilai lebih rendah menunjukkan kesamaan bentuk yang lebih baik.
#                     - **Tangent-Space Mean Distance (TMD)**: Mengukur jarak rata-rata pada ruang tangensial. Nilai lebih rendah menunjukkan kesamaan bentuk lokal yang lebih baik.
#                     - **Chamfer Distance (CD)**: Mengukur jarak rata-rata antar titik. Nilai lebih rendah menunjukkan kecocokan bentuk yang lebih baik.
#                     - **IoU Score**: Mengukur volume tumpang tindih. Nilai lebih tinggi (0-1) menunjukkan kesamaan volume yang lebih baik.
                    
#                     Untuk metrik evaluasi yang akurat, unggah model referensi.
#                     """)
    
#     with gr.Row(variant="panel"):
#         gr.Examples(
#             examples=[
#                 "examples/garuda-wisnu-kencana.png",
#                 "examples/tapel-barong1.png",
#                 "examples/tapel-barong2.png",
#                 "examples/pintu-belok.png",
#             ],
#             inputs=[input_image],
#             outputs=[processed_image, output_model_obj, output_model_glb, f1_metric, uhd_metric, tmd_metric, cd_metric, iou_metric, metrics_text, radar_plot, bar_plot],
#             cache_examples=False,
#             fn=partial(run_example),
#             label="Contoh",
#             examples_per_page=20,
#         )
    
#     # Create a popup for evaluation metrics info
#     evaluation_info_md = gr.Markdown(visible=False)
    
#     def show_evaluation_info():
#         return evaluation_info_md.update(visible=True), evaluation_info.update(visible=False)
    
#     evaluation_info.click(
#         fn=show_evaluation_info,
#         inputs=[],
#         outputs=[evaluation_info_md],
#     )
        
#     submit.click(fn=check_input_image, inputs=[input_image]).success(
#         fn=preprocess,
#         inputs=[input_image, do_remove_background, foreground_ratio],
#         outputs=[processed_image],
#     ).success(
#         fn=lambda img, rb, fr, mc, rm, mq, tq, sf: 
#             generate(
#                 preprocess(img, rb, fr),
#                 mc, rm, ["obj", "glb"], mq, tq, sf
#             ),
#         inputs=[
#             input_image, 
#             do_remove_background, 
#             foreground_ratio,
#             mc_resolution,
#             reference_model,
#             model_quality,
#             texture_quality,
#             smoothing_factor
#         ],
#         outputs=[
#             output_model_obj, 
#             output_model_glb,
#             f1_metric,
#             uhd_metric,
#             tmd_metric,
#             cd_metric,
#             iou_metric,
#             metrics_text,
#             radar_plot,
#             bar_plot
#         ]
#     )



# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--username', type=str, default=None, help='Username for authentication')
#     parser.add_argument('--password', type=str, default=None, help='Password for authentication')
#     parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
#     parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
#     parser.add_argument("--share", action='store_true', help="make the UI accessible through gradio.live")
#     parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    
#     args = parser.parse_args()
    
#     # Configure queue before launch
#     interface.queue(max_size=args.queuesize)
    
#     # Prepare auth tuple
#     auth = None
#     if args.username and args.password:
#         auth = (args.username, args.password)
    
#     # Launch with simplified parameters
#     try:
#         interface.launch(
#             server_port=args.port,
#             server_name="0.0.0.0" if args.listen else None,
#             share=args.share,
#             auth=auth,
#             debug=True  # Add debug mode to see more detailed errors
#         )
#     except Exception as e:
#         print(f"Failed to launch interface: {str(e)}")
#         # Fallback to basic launch if custom configuration fails
#         interface.launch()



import logging
import os
import tempfile
import time

import gradio as gr
import numpy as np
import rembg
import torch
import trimesh
from PIL import Image
from functools import partial
import plotly.graph_objects as go
import plotly.express as px
import json

from tsr.system import TSR
from tsr.utils import remove_background, resize_foreground, to_gradio_3d_orientation
from tsr.evaluation import calculate_metrics

import argparse


# Configure CUDA memory settings
if torch.cuda.is_available():
    device = "cuda:0"
    # Lower default chunk size to reduce memory usage
    default_chunk_size = 8192
else:
    device = "cpu"
    default_chunk_size = 8192

# Load both models
print("Loading original TripoSR model...")
model_original = TSR.from_pretrained(
    "stabilityai/TripoSR",
    config_name="config.yaml",
    weight_name="model.ckpt",
)

print("Loading custom TripoSR model...")
model_custom = TSR.from_pretrained(
    "TrianC0de/TripoSR2",
    config_name="config.yaml",
    weight_name="sf3d_checkpoint_epoch_5000.ckpt", 
)

# Configure both models
model_original.renderer.set_chunk_size(default_chunk_size)
model_original.to(device)

model_custom.renderer.set_chunk_size(default_chunk_size)
model_custom.to(device)

print("Both models loaded successfully!")

rembg_session = rembg.new_session()

# Global storage for historical metrics to enable comparison
metrics_history = []

def ensemble_meshes(mesh1, mesh2, blend_method="weighted_average", weight1=0.5, weight2=0.5):
    """
    Combine two meshes using different blending methods
    """
    if blend_method == "weighted_average":
        # Weighted average of vertices
        if len(mesh1.vertices) == len(mesh2.vertices):
            # If same vertex count, direct weighted average
            new_vertices = weight1 * mesh1.vertices + weight2 * mesh2.vertices
            # Use faces from the first mesh (assuming same topology)
            combined_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh1.faces)
        else:
            # Different vertex counts - use spatial blending
            combined_mesh = spatial_blend_meshes(mesh1, mesh2, weight1, weight2)
            
    elif blend_method == "detail_fusion":
        # Use high-frequency details from one, low-frequency from another
        combined_mesh = detail_fusion_meshes(mesh1, mesh2, weight1)
        
    elif blend_method == "vertex_density_blend":
        # Blend based on vertex density
        combined_mesh = density_blend_meshes(mesh1, mesh2)
        
    else:  # "concatenate"
        # Simple concatenation (might create artifacts)
        combined_vertices = np.concatenate([mesh1.vertices, mesh2.vertices + [2, 0, 0]], axis=0)
        combined_faces = np.concatenate([mesh1.faces, mesh2.faces + len(mesh1.vertices)], axis=0)
        combined_mesh = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces)
    
    # Clean up the resulting mesh
    combined_mesh.remove_duplicate_faces()
    combined_mesh.remove_unreferenced_vertices()
    combined_mesh.fix_normals()
    
    return combined_mesh

def spatial_blend_meshes(mesh1, mesh2, weight1, weight2):
    """
    Blend meshes by creating a new mesh that combines spatial information.
    This version is robust against different trimesh API versions.
    """
    n_samples = 10000

    # --- Defensive sampling for mesh1 ---
    sample1_result = mesh1.sample(n_samples)
    # Check if the result is a tuple (like (points, faces))
    if isinstance(sample1_result, tuple):
        # If yes, take the first element which is the points
        points1 = sample1_result[0]
    else:
        # If no, assume the result is the points array itself
        points1 = sample1_result

    # --- Defensive sampling for mesh2 ---
    sample2_result = mesh2.sample(n_samples)
    if isinstance(sample2_result, tuple):
        points2 = sample2_result[0]
    else:
        points2 = sample2_result
    
    # Combine sampled points with weights
    combined_points = np.concatenate([
        points1 * weight1,
        points2 * weight2
    ])
    
    # Create a new mesh using marching cubes or convex hull
    try:
        # Use convex hull as a simple approach
        combined_mesh = trimesh.convex.convex_hull(combined_points)
    except:
        # Fallback: use alpha shape or return first mesh
        combined_mesh = mesh1
    
    return combined_mesh

def detail_fusion_meshes(mesh1, mesh2, detail_weight=0.7):
    """
    Fuse details from two meshes - use base shape from one, details from another
    """
    # Use mesh1 as base, add high-frequency details from mesh2
    if len(mesh1.vertices) == len(mesh2.vertices):
        # Calculate displacement between meshes
        displacement = mesh2.vertices - mesh1.vertices
        # Apply scaled displacement
        new_vertices = mesh1.vertices + displacement * detail_weight
        combined_mesh = trimesh.Trimesh(vertices=new_vertices, faces=mesh1.faces)
    else:
        # Different topology - use weighted spatial blend
        combined_mesh = spatial_blend_meshes(mesh1, mesh2, 1-detail_weight, detail_weight)
    
    return combined_mesh

def density_blend_meshes(mesh1, mesh2):
    """
    Blend meshes based on vertex density - keep denser regions
    """
    # Sample more points from denser areas
    n_samples = 15000
    points1, _ = mesh1.sample(n_samples)
    points2, _ = mesh2.sample(n_samples)
    
    # Combine all points
    all_points = np.concatenate([points1, points2])
    
    # Create new mesh from combined point cloud
    try:
        combined_mesh = trimesh.convex.convex_hull(all_points)
    except:
        combined_mesh = mesh1
    
    return combined_mesh

def create_metrics_radar_chart(current_metrics):
    """Create a radar chart comparing the current metrics with historical averages"""
    metrics_to_show = {
        'f1_score': {'display': 'F1', 'invert': True},  # <-- Perubahan di sini
        'uniform_hausdorff_distance': {'display': 'UHD', 'invert': True},
        'tangent_space_mean_distance': {'display': 'TMD', 'invert': True},
        'chamfer_distance': {'display': 'CD', 'invert': True},
        'iou_score': {'display': 'IoU', 'invert': False}
    }
    # metrics_to_show = {
    #     'f1_score': {'display': 'F1', 'invert': True},
    #     'uniform_hausdorff_distance': {'display': 'UHD', 'invert': True},
    #     'tangent_space_mean_distance': {'display': 'TMD', 'invert': True},
    #     'chamfer_distance': {'display': 'CD', 'invert': True},
    #     'iou_score': {'display': 'IoU', 'invert': False}
    # }
    
    available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
    if not available_metrics or len(metrics_history) == 0:
        fig = go.Figure()
        fig.add_annotation(
            text="Generate more models to see comparison with historical average",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Metrics Comparison")
        return fig
    
    # Calculate average of historical metrics
    avg_metrics = {}
    for metric_name in available_metrics.keys():
        valid_hist = [hist for hist in metrics_history if metric_name in hist]
        if valid_hist:
            avg_metrics[metric_name] = sum(hist[metric_name] for hist in valid_hist) / len(valid_hist)
        else:
            avg_metrics[metric_name] = current_metrics[metric_name]
    
    categories = [available_metrics[m]['display'] for m in available_metrics.keys()]
    
    current_values = []
    history_values = []
    
    for metric_name, config in available_metrics.items():
        current_val = current_metrics[metric_name]
        avg_val = avg_metrics[metric_name]

        if config['invert']:
            max_val = max(current_val, avg_val) * 1.2
            denominator = max_val + 1e-9
            current_values.append(1 - (current_val / denominator))
            history_values.append(1 - (avg_val / denominator))

        else:
            current_values.append(current_val)
            history_values.append(avg_val)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=current_values,
        theta=categories,
        fill='toself',
        name='Current Model'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=history_values,
        theta=categories,
        fill='toself',
        name='Historical Average'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        title="Metrics Comparison (Higher is Better)"
    )
    
    return fig
# GANTI TOTAL FUNGSI create_metrics_bar_chart ANDA DENGAN YANG DI BAWAH INI

def create_metrics_bar_chart(current_metrics):
    """Create a bar chart for current metrics"""
    metrics_to_show = {
        'f1_score': {'display': 'F1 Score (↑)', 'color': 'purple'},
        'uniform_hausdorff_distance': {'display': 'UHD (↓)', 'color': 'red'},
        'tangent_space_mean_distance': {'display': 'TMD (↓)', 'color': 'orange'},
        'chamfer_distance': {'display': 'CD (↓)', 'color': 'green'},
        'iou_score': {'display': 'IoU (↑)', 'color': 'blue'}
    }
    
    available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
    if not available_metrics:
        fig = go.Figure()
        fig.add_annotation(
            text="No metrics available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False
        )
        fig.update_layout(title="Metrics")
        return fig
    
    names = [available_metrics[m]['display'] for m in available_metrics.keys()]
    values = []
    colors = [available_metrics[m]['color'] for m in available_metrics.keys()]

    # --- PERUBAHAN UTAMA ADA DI SINI ---
    for metric_name in available_metrics.keys():
        # Jika metriknya adalah f1_score, hitung 1 - nilainya
        if metric_name == 'f1_score':
            f1_error = 1.0 - float(current_metrics[metric_name])
            values.append(f1_error)
        # Untuk metrik lain, gunakan nilai aslinya
        else:
            values.append(current_metrics[metric_name])
    # ------------------------------------

    fig = go.Figure(data=[
        go.Bar(
            x=names,
            y=values,
            marker_color=colors
        )
    ])
    
    fig.update_layout(
        title="Current Metrics",
        xaxis_title="Metric",
        yaxis_title="Value",
        yaxis=dict(
            title="Value",
            titlefont_size=16,
            tickfont_size=14,
        )
    )
    
    return fig
    
# def create_metrics_bar_chart(current_metrics):
#     """Create a bar chart for current metrics"""
#     metrics_to_show = {
#         'f1_score': {'display': 'F1 Score (↑)', 'color': 'purple', 'invert': True},
#         'uniform_hausdorff_distance': {'display': 'UHD (↓)', 'color': 'red'},
#         'tangent_space_mean_distance': {'display': 'TMD (↓)', 'color': 'orange'},
#         'chamfer_distance': {'display': 'CD (↓)', 'color': 'green'},
#         'iou_score': {'display': 'IoU (↑)', 'color': 'blue'}
#     }
    
#     available_metrics = {k: v for k, v in metrics_to_show.items() if k in current_metrics}
    
#     if not available_metrics:
#         fig = go.Figure()
#         fig.add_annotation(
#             text="No metrics available",
#             xref="paper", yref="paper",
#             x=0.5, y=0.5,
#             showarrow=False
#         )
#         fig.update_layout(title="Metrics")
#         return fig
    
#     names = [available_metrics[m]['display'] for m in available_metrics.keys()]
#     values = [current_metrics[m] for m in available_metrics.keys()]
#     colors = [available_metrics[m]['color'] for m in available_metrics.keys()]
    
#     fig = go.Figure(data=[
#         go.Bar(
#             x=names,
#             y=values,
#             marker_color=colors
#         )
#     ])
    
#     fig.update_layout(
#         title="Current Metrics",
#         xaxis_title="Metric",
#         yaxis_title="Value",
#         yaxis=dict(
#             title="Value",
#             titlefont_size=16,
#             tickfont_size=14,
#         )
#     )
    
#     return fig

def check_input_image(input_image):
    if input_image is None:
        raise gr.Error("No image uploaded!")

def preprocess(input_image, do_remove_background, foreground_ratio):
    def fill_background(image):
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        return Image.fromarray((image * 255.0).astype(np.uint8))

    if do_remove_background:
        image = input_image.convert("RGB")
        image = remove_background(image, rembg_session)
        image = resize_foreground(image, foreground_ratio)
        image = fill_background(image)
    else:
        image = input_image
        if image.mode == "RGBA":
            image = fill_background(image)
            
    # Ensure image size is reasonable
    max_size = 512
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        
    return image
def create_point_cloud_mesh(mesh, radius=0.005):
    """
    Membuat representasi visual point cloud dengan membuat bola kecil di setiap vertex.
    Mengembalikan sebuah objek Trimesh tunggal.
    """
    # Coba dapatkan warna, jika tidak ada, beri warna abu-abu
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
        vertex_colors = mesh.visual.vertex_colors
    else:
        num_vertices = len(mesh.vertices)
        vertex_colors = np.full((num_vertices, 4), [200, 200, 200, 255], dtype=np.uint8)

    # Buat daftar untuk menampung semua bola kecil
    spheres = []
    for i, vertex in enumerate(mesh.vertices):
        # Buat bola kecil di posisi vertex
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        sphere.vertices += vertex
        
        # Beri warna pada bola sesuai warna vertex
        sphere.visual.vertex_colors = vertex_colors[i]
        
        spheres.append(sphere)
    
    # Gabungkan semua bola kecil menjadi satu mesh besar
    if not spheres:
        return trimesh.Trimesh() # Kembalikan mesh kosong jika tidak ada vertices
        
    combined_mesh = trimesh.util.concatenate(spheres)
    return combined_mesh
    
def fix_model_orientation(mesh):
    """Fix the orientation of the model for proper display"""
    # Rotate 90 degrees around X axis to match standard orientation
    rotation_matrix = trimesh.transformations.rotation_matrix(
        angle=np.pi/2,
        direction=[1, 0, 0],
        point=[0, 0, 0]
    )
    mesh.apply_transform(rotation_matrix)
    
    # Center the mesh
    mesh.vertices -= mesh.vertices.mean(axis=0)
    
    # Scale to fit in a unit cube
    max_extent = max(mesh.vertices.max(axis=0) - mesh.vertices.min(axis=0))
# Only scale if the mesh has a non-zero size
    if max_extent > 1e-6:  # Using a small threshold for float comparison
        scale = 1.0 / max_extent
        mesh.vertices *= scale
    
    # Fix normals to ensure proper rendering
    mesh.fix_normals()
    
    # Ensure material properties aren't too reflective
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        if hasattr(mesh.visual.material, 'specular'):
            mesh.visual.material.specular = [0.1, 0.1, 0.1, 1.0]
        if hasattr(mesh.visual.material, 'ambient'):
            mesh.visual.material.ambient = [0.6, 0.6, 0.6, 1.0]
        if hasattr(mesh.visual.material, 'shininess'):
            mesh.visual.material.shininess = 0.1

    return mesh

# GANTI TOTAL FUNGSI GENERATE ANDA DENGAN VERSI FINAL DI BAWAH INI

def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb", "ply"], 
             model_quality="Standar", texture_quality=7, smoothing_factor=0.3,
             use_model="Both", blend_method="weighted_average", model_weight=0.5):
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        output_dir = os.path.join(os.getcwd(), "outputs")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        quality_settings = {
            "Konsep": {"chunk_size": 32768, "detail_factor": 0.5},
            "Standar": {"chunk_size": 16384, "detail_factor": 0.7},
            "Tinggi": {"chunk_size": 8192, "detail_factor": 1.0}
        }
        
        chunk_size = quality_settings[model_quality]["chunk_size"]
        model_original.renderer.set_chunk_size(chunk_size)
        model_custom.renderer.set_chunk_size(chunk_size)
        
        with torch.inference_mode():
            if use_model == "Original Only":
                scene_codes = model_original(image, device=device)
                mesh = model_original.extract_mesh(
                    scene_codes, True, resolution=min(mc_resolution, 192)
                )[0]
            elif use_model == "Custom Only":
                scene_codes = model_custom(image, device=device)
                mesh = model_custom.extract_mesh(
                    scene_codes, True, resolution=min(mc_resolution, 192)
                )[0]
            else:
                scene_codes_original = model_original(image, device=device)
                mesh_original = model_original.extract_mesh(
                    scene_codes_original, True, resolution=min(mc_resolution, 192)
                )[0]
                scene_codes_custom = model_custom(image, device=device)
                mesh_custom = model_custom.extract_mesh(
                    scene_codes_custom, True, resolution=min(mc_resolution, 192)
                )[0]
                mesh = ensemble_meshes(
                    mesh_original, mesh_custom, blend_method=blend_method,
                    weight1=model_weight, weight2=(1.0 - model_weight)
                )
        
        mesh = to_gradio_3d_orientation(mesh)
        mesh = fix_model_orientation(mesh)
        
        if smoothing_factor > 0 and len(mesh.vertices) > 0:
            from trimesh import smoothing
            iterations = max(1, int(smoothing_factor * 10))
            smoothing.filter_laplacian(mesh, iterations=iterations)
        
        if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
            colors = mesh.visual.vertex_colors
            import colorsys
            normalized_colors = np.zeros_like(colors)
            for i in range(len(colors)):
                r, g, b = colors[i][0]/255.0, colors[i][1]/255.0, colors[i][2]/255.0
                h, s, v = colorsys.rgb_to_hsv(r, g, b)
                v = min(v, 0.95)
                s = min(s * 1.1, 1.0)
                r, g, b = colorsys.hsv_to_rgb(h, s, v)
                normalized_colors[i][0] = int(r * 255)
                normalized_colors[i][1] = int(g * 255)
                normalized_colors[i][2] = int(b * 255)
                normalized_colors[i][3] = colors[i][3]
            mesh.visual.vertex_colors = normalized_colors
            
        reference_mesh = None
        if reference_model is not None:
            reference_mesh = trimesh.load(reference_model.name)
        
        metrics = calculate_metrics(mesh, reference_mesh)
        
        global metrics_history
        metrics_history.append(metrics)
        if len(metrics_history) > 10:
            metrics_history = metrics_history[-10:]
            
        radar_chart = create_metrics_radar_chart(metrics)
        bar_chart = create_metrics_bar_chart(metrics)
        
        model_info = f"Model used: {use_model}"
        if use_model == "Both":
            model_info += f" (Original: {model_weight:.1f}, Custom: {1-model_weight:.1f}, Method: {blend_method})"
        
        metrics_text = f"{model_info}\n\nMetrics:\n"
        if 'f1_score' in metrics:
            f1_error = 1.0 - float(metrics['f1_score'])
            metrics_text += f"F1 Error (1-F1): {f1_error:.4f}\n"
        if 'uniform_hausdorff_distance' in metrics: metrics_text += f"UHD: {metrics['uniform_hausdorff_distance']:.4f}\n"
        if 'tangent_space_mean_distance' in metrics: metrics_text += f"TMD: {metrics['tangent_space_mean_distance']:.4f}\n"
        if 'chamfer_distance' in metrics: metrics_text += f"CD: {metrics['chamfer_distance']:.4f}\n"
        if 'iou_score' in metrics: metrics_text += f"IoU Score: {metrics.get('iou_score', metrics.get('iou', 0.0)):.4f}"
        if reference_mesh is None: metrics_text += "\nNote: For more accurate metrics, provide a reference model."
        
        # --- PERUBAHAN LOGIKA PENYIMPANAN FILE ---
        
        rv = []
        # Buat representasi mesh dari point cloud
        point_cloud_as_mesh = create_point_cloud_mesh(mesh)

        for fmt in formats:
            file_path = os.path.join(output_dir, f"model_{use_model.replace(' ', '_')}_{timestamp}.{fmt}")
            if fmt == "ply":
                # Ekspor "point cloud" kita sebagai file OBJ agar pasti bisa dibaca
                point_cloud_as_mesh.export(file_path.replace('.ply', '.obj'))
                rv.append(file_path.replace('.ply', '.obj'))
            elif fmt in ['obj', 'glb']:
                mesh.export(file_path)
                rv.append(file_path)
        
        rv.extend([
            (1.0 - float(metrics.get("f1_score", 0.0))),
            float(metrics.get("uniform_hausdorff_distance", 0.0)),
            float(metrics.get("tangent_space_mean_distance", 0.0)),
            float(metrics.get("chamfer_distance", 0.0)),
            float(metrics.get("iou_score", metrics.get("iou", 0.0))),
            metrics_text,
            radar_chart,
            bar_chart
        ])
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return rv
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            raise gr.Error("GPU memory error. Try 'Konsep' quality or lower resolution.")
        else:
            raise gr.Error(f"Generation error: {str(e)}")
    except Exception as e:
        raise gr.Error(f"Error: {str(e)}")
        
# def generate(image, mc_resolution, reference_model=None, formats=["obj", "glb", "ply"], 
#              model_quality="Standar", texture_quality=7, smoothing_factor=0.3,
#              use_model="Both", blend_method="weighted_average", model_weight=0.5):
#     try:
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         output_dir = os.path.join(os.getcwd(), "outputs")
#         os.makedirs(output_dir, exist_ok=True)
#         timestamp = time.strftime("%Y%m%d_%H%M%S")
        
#         quality_settings = {
#             "Konsep": {"chunk_size": 32768, "detail_factor": 0.5},
#             "Standar": {"chunk_size": 16384, "detail_factor": 0.7},
#             "Tinggi": {"chunk_size": 8192, "detail_factor": 1.0}
#         }
        
#         chunk_size = quality_settings[model_quality]["chunk_size"]
#         model_original.renderer.set_chunk_size(chunk_size)
#         model_custom.renderer.set_chunk_size(chunk_size)
        
#         with torch.inference_mode():
#             # FIX #1: Memanggil extract_mesh sesuai versi library Anda
#             if use_model == "Original Only":
#                 scene_codes = model_original(image, device=device)
#                 mesh = model_original.extract_mesh(
#                     scene_codes, 
#                     True, # Ini untuk has_vertex_color
#                     resolution=min(mc_resolution, 192)
#                 )[0]
                
#             elif use_model == "Custom Only":
#                 scene_codes = model_custom(image, device=device)
#                 mesh = model_custom.extract_mesh(
#                     scene_codes, 
#                     True, # Ini untuk has_vertex_color
#                     resolution=min(mc_resolution, 192)
#                 )[0]

#             else:
#                 scene_codes_original = model_original(image, device=device)
#                 mesh_original = model_original.extract_mesh(
#                     scene_codes_original, 
#                     True, # Ini untuk has_vertex_color
#                     resolution=min(mc_resolution, 192)
#                 )[0]
                
#                 scene_codes_custom = model_custom(image, device=device)
#                 mesh_custom = model_custom.extract_mesh(
#                     scene_codes_custom, 
#                     True, # Ini untuk has_vertex_color
#                     resolution=min(mc_resolution, 192)
#                 )[0]
                
#                 mesh = ensemble_meshes(
#                     mesh_original, mesh_custom, 
#                     blend_method=blend_method,
#                     weight1=model_weight, weight2=(1.0 - model_weight)
#                 )
        
#         mesh = to_gradio_3d_orientation(mesh)
#         mesh = fix_model_orientation(mesh)
        
#         if smoothing_factor > 0:
#             if hasattr(mesh, 'vertices') and len(mesh.vertices) > 0:
#                 from trimesh import smoothing
#                 iterations = max(1, int(smoothing_factor * 10))
#                 smoothing.filter_laplacian(mesh, iterations=iterations)
        
#         if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
#             colors = mesh.visual.vertex_colors
#             import colorsys
#             normalized_colors = np.zeros_like(colors)
#             for i in range(len(colors)):
#                 r, g, b = colors[i][0]/255.0, colors[i][1]/255.0, colors[i][2]/255.0
#                 h, s, v = colorsys.rgb_to_hsv(r, g, b)
#                 v = min(v, 0.95)
#                 s = min(s * 1.1, 1.0)
#                 r, g, b = colorsys.hsv_to_rgb(h, s, v)
#                 normalized_colors[i][0] = int(r * 255)
#                 normalized_colors[i][1] = int(g * 255)
#                 normalized_colors[i][2] = int(b * 255)
#                 normalized_colors[i][3] = colors[i][3]
#             mesh.visual.vertex_colors = normalized_colors
            
#         reference_mesh = None
#         if reference_model is not None:
#             reference_mesh = trimesh.load(reference_model.name)
            
#         metrics = calculate_metrics(mesh, reference_mesh)
        
#         global metrics_history
#         metrics_history.append(metrics)
#         if len(metrics_history) > 10:
#             metrics_history = metrics_history[-10:]
            
#         radar_chart = create_metrics_radar_chart(metrics)
#         bar_chart = create_metrics_bar_chart(metrics)
        
#         model_info = f"Model used: {use_model}"
#         if use_model == "Both":
#             model_info += f" (Original: {model_weight:.1f}, Custom: {1-model_weight:.1f}, Method: {blend_method})"
        
#         metrics_text = f"{model_info}\n\nMetrics:\n"
#         if 'f1_score' in metrics:
#             f1_error = 1.0 - float(metrics['f1_score'])
#             metrics_text += f"F1 Score: {f1_error:.4f}\n"
#         if 'uniform_hausdorff_distance' in metrics: metrics_text += f"UHD: {metrics['uniform_hausdorff_distance']:.4f}\n"
#         if 'tangent_space_mean_distance' in metrics: metrics_text += f"TMD: {metrics['tangent_space_mean_distance']:.4f}\n"
#         if 'chamfer_distance' in metrics: metrics_text += f"CD: {metrics['chamfer_distance']:.4f}\n"
#         if 'iou_score' in metrics: metrics_text += f"IoU Score: {metrics.get('iou_score', metrics.get('iou', 0.0)):.4f}"
#         if reference_mesh is None: metrics_text += "\nNote: For more accurate metrics, provide a reference model."


#         print("--- DEBUGGING MESH ---")
#         print(f"Tipe objek mesh: {type(mesh)}")
#         print(f"Jumlah vertices: {len(mesh.vertices)}")
#         print(f"Apakah punya warna?: {hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors')}")
#         print("--- AKHIR DEBUG ---")
        
#         vertex_colors = None
        
#         # FIX #2: Membuat point_cloud SEBELUM loop penyimpanan
#         print("--- DEBUGGING LANJUTAN ---")
#         print("Memaksa semua titik point cloud menjadi MERAH.")
#         num_vertices = len(mesh.vertices)
#         forced_colors = np.full((num_vertices, 4), [255, 0, 0, 255], dtype=np.uint8)
        
#         point_cloud = trimesh.points.PointCloud(mesh.vertices, colors=forced_colors)
#         print("--- AKHIR DEBUG LANJUTAN ---")

#         # vertex_colors = None
#         # if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'vertex_colors'):
#         #     vertex_colors = mesh.visual.vertex_colors
#         # else:
#         #     num_vertices = len(mesh.vertices)
#         #     vertex_colors = np.full((num_vertices, 4), [200, 200, 200, 255], dtype=np.uint8)
#         # point_cloud = trimesh.points.PointCloud(mesh.vertices, colors=vertex_colors)
        
#         rv = []
#         for format in formats:
#             file_path = os.path.join(output_dir, f"model_{use_model.replace(' ', '_')}_{timestamp}.{format}")
#             if format == "ply":
#                 point_cloud.export(file_path)
#             else:
#                 mesh.export(file_path)
#             rv.append(file_path)
        
#         # FIX #3: Membungkus semua metrik dengan float() untuk mencegah TypeError
#         rv.extend([
#             (1.0 - float(metrics.get("f1_score", 0.0))),
#             float(metrics.get("uniform_hausdorff_distance", 0.0)),
#             float(metrics.get("tangent_space_mean_distance", 0.0)),
#             float(metrics.get("chamfer_distance", 0.0)),
#             float(metrics.get("iou_score", metrics.get("iou", 0.0))),
#             metrics_text,
#             radar_chart,
#             bar_chart
#         ])
        
#         if torch.cuda.is_available():
#             torch.cuda.empty_cache()
            
#         return rv
#     except RuntimeError as e:
#         if "CUDA out of memory" in str(e):
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()
#             raise gr.Error("GPU memory error. Try 'Konsep' quality or lower resolution.")
#         else:
#             raise gr.Error(f"Generation error: {str(e)}")
#     except Exception as e:
#         raise gr.Error(f"Error: {str(e)}")

def run_example(image_pil):
    preprocessed = preprocess(image_pil, False, 0.9)
    # Panggil generate untuk mendapatkan semua 11 hasil
    results = generate(
        preprocessed, 128, None, ["obj", "glb", "ply"],
        "Standar", 7, 0.3, "Both", "weighted_average", 0.5
    )
    return [preprocessed] + results

def run_generation_pipeline(
    input_image, do_remove_background, foreground_ratio, 
    mc_resolution, reference_model, model_quality, 
    texture_quality, smoothing_factor, use_model, 
    blend_method, model_weight
):
    """
    A single function to handle the complete generation process from a Gradio button click.
    """
    # 1. Input Validation (raises gr.Error on failure, stopping execution)
    check_input_image(input_image)

    # 2. Preprocess the image once
    processed_image = preprocess(input_image, do_remove_background, foreground_ratio)

    # 3. Generate the 3D model and metrics. 
    # The 'generate' function returns a list of 10 items.
    generation_results = generate(
        processed_image,
        mc_resolution,
        reference_model,
        ["obj", "glb", "ply"],
        model_quality,
        texture_quality,
        smoothing_factor,
        use_model,
        blend_method,
        model_weight
    )

    return [processed_image] + generation_results

with gr.Blocks(title="Dual Model 3D Generation") as interface:
    gr.Markdown(
        """    
# Generasi Model 3D dari Gambar (Dual Model)

Unggah gambar untuk menghasilkan model 3D menggunakan model original TripoSR, model kustom, atau kombinasi keduanya.

## Model Selection & Ensemble Parameters

- **Model Selection**: Pilih model yang akan digunakan:
  - **Original Only**: Hanya menggunakan model TripoSR original dari Stability AI
  - **Custom Only**: Hanya menggunakan model kustom yang sudah di-fine-tune
  - **Both**: Menggunakan ensemble (gabungan) dari kedua model untuk hasil terbaik

- **Model Weight**: Ketika menggunakan "Both", mengontrol kontribusi model original vs kustom (0.0 = 100% kustom, 1.0 = 100% original)

- **Blend Method**: Metode penggabungan ketika menggunakan "Both":
  - **Weighted Average**: Rata-rata berbobot dari kedua mesh
  - **Detail Fusion**: Menggunakan bentuk dasar dari satu model, detail dari model lain
  - **Vertex Density Blend**: Menggabungkan berdasarkan kepadatan vertex
  - **Concatenate**: Penggabungan sederhana (mungkin menghasilkan artifact)

## Fine-Tuning Parameters

- **Foreground Ratio**: Mengontrol seberapa banyak gambar yang dianggap sebagai latar depan saat pemrosesan.
- **Marching Cubes Resolution**: Mengontrol tingkat detail mesh 3D.
- **Kualitas Model**: Draft/Standar/Tinggi - mempengaruhi waktu pemrosesan dan detail hasil.
- **Kualitas Tekstur**: Mengontrol detail tekstur yang diterapkan pada model.
- **Mesh Smoothing**: Menerapkan penghalusan pada model akhir.

## Tips Penggunaan Dual Model:
1. **Both + Weighted Average (0.5)** biasanya memberikan hasil terbaik untuk sebagian besar kasus
2. **Detail Fusion** cocok jika satu model bagus untuk bentuk umum, yang lain untuk detail
3. Sesuaikan **Model Weight** berdasarkan kekuatan masing-masing model pada jenis gambar tertentu
4. Model ensemble membutuhkan lebih banyak memori GPU - gunakan kualitas lebih rendah jika mengalami error memori
    """
    )
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_image = gr.Image(
                    label="Gambar Input",
                    image_mode="RGBA",
                    sources="upload",
                    type="pil",
                    elem_id="content_image",
                )
                processed_image = gr.Image(label="Processed Image", interactive=False)
            with gr.Row():
                with gr.Group():
                    # Model selection parameters
                    use_model = gr.Dropdown(
                        ["Original Only", "Custom Only", "Both"],
                        value="Both",
                        label="Model Selection",
                        info="Choose which model(s) to use"
                    )
                    
                    with gr.Row():
                        model_weight = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="Model Weight (Original ← → Custom)",
                            info="0.0 = 100% Custom, 1.0 = 100% Original"
                        )
                    
                    blend_method = gr.Dropdown(
                        ["weighted_average", "detail_fusion", "vertex_density_blend", "concatenate"],
                        value="weighted_average",
                        label="Blend Method",
                        info="How to combine models when using 'Both'"
                    )
                    
                    gr.Markdown("---")
                    
                    # Original parameters
                    do_remove_background = gr.Checkbox(
                        label="Hapus Latar Belakang", value=True
                    )
                    foreground_ratio = gr.Slider(
                        minimum=0.5,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Foreground Ratio",
                    )
                    mc_resolution = gr.Slider(
                        minimum=64,
                        maximum=256,
                        value=128,
                        step=16,
                        label="Resolusi Marching Cubes",
                    )
                    model_quality = gr.Dropdown(
                        ["Konsep", "Standar", "Tinggi"],
                        value="Standar",
                        label="Model Quality",
                    )
                    texture_quality = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=7,
                        step=1,
                        label="Kualitas Tekstur",
                    )
                    smoothing_factor = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.3,
                        step=0.1,
                        label="Mesh Smoothing",
                    )
                    reference_model = gr.File(
                        label="Reference Model (OBJ/GLB/STL) [optional]", 
                        file_types=[".obj", ".glb", ".stl"]
                    )
                    submit = gr.Button("Generate 3D Model", variant="primary")
        
        with gr.Column():
            with gr.Tabs():
                with gr.TabItem("3D Visualization"):
                    output_model_obj = gr.Model3D(
                        label="Model 3D (OBJ)",
                        interactive=False
                    )
                    output_model_glb = gr.Model3D(
                        label="Model 3D (GLB)",
                        interactive=False
                    )
                    output_model_ply = gr.Model3D(
                        label="Point Cloud (PLY)",
                        interactive=False
                    )
                    
                with gr.TabItem("Metrik Evaluasi"):
                    with gr.Row():
                        f1_metric = gr.Number(label="F1 Score", value=0.0, precision=4)
                        uhd_metric = gr.Number(label="Uniform Hausdorff Distance", value=0.0, precision=4)
                        tmd_metric = gr.Number(label="Tangent-Space Mean Distance", value=0.0, precision=4)
                        cd_metric = gr.Number(label="Chamfer Distance", value=0.0, precision=4)
                        iou_metric = gr.Number(label="IoU Score", value=0.0, precision=4)
                    
                    with gr.Row():
                        metrics_text = gr.Textbox(
                            label="Metrik Lengkap", 
                            value="Hasilkan model untuk melihat metrik evaluasi.\n\nUntuk perbandingan yang lebih akurat, unggah model referensi.",
                            lines=8
                        )
                
                with gr.TabItem("Visualisasi Metrik"):
                    gr.Markdown("""
                    ### Visualisasi Perbandingan Metrik
                    
                    Diagram di bawah menunjukkan perbandingan metrik model saat ini dengan rata-rata historis.
                    Semakin besar nilai pada diagram radar, semakin baik kualitas metrik tersebut.
                    """)
                    with gr.Row():
                        radar_plot = gr.Plot(label="Perbandingan dengan Riwayat", show_label=False)
                    
                    gr.Markdown("""
                    ### Nilai Metrik Saat Ini
                    
                    Diagram batang di bawah menunjukkan nilai absolut dari metrik saat ini.
                    UHD, TMD, CD: nilai lebih rendah lebih baik (↓)
                    F1, IoU: nilai lebih tinggi lebih baik (↑)
                    """)
                    with gr.Row():
                        bar_plot = gr.Plot(label="Nilai Metrik Saat Ini", show_label=False)
                    
                    gr.Markdown("""
                    **Petunjuk Metrik:**
                    - **F1 Score**: Mengukur keseimbangan antara presisi dan recall. Nilai lebih tinggi (0-1) menunjukkan kecocokan permukaan yang lebih baik.
                    - **Uniform Hausdorff Distance (UHD)**: Mengukur jarak maksimum antara permukaan mesh. Nilai lebih rendah menunjukkan kesamaan bentuk yang lebih baik.
                    - **Tangent-Space Mean Distance (TMD)**: Mengukur jarak rata-rata pada ruang tangensial. Nilai lebih rendah menunjukkan kesamaan bentuk lokal yang lebih baik.
                    - **Chamfer Distance (CD)**: Mengukur jarak rata-rata antar titik. Nilai lebih rendah menunjukkan kecocokan bentuk yang lebih baik.
                    - **IoU Score**: Mengukur volume tumpang tindih. Nilai lebih tinggi (0-1) menunjukkan kesamaan volume yang lebih baik.
                    
                    **Ensemble Model Performance:**
                    - **Original Only**: Menggunakan model pre-trained Stability AI (stabil, konsisten)
                    - **Custom Only**: Menggunakan model fine-tuned Anda (spesialisasi domain tertentu)
                    - **Both**: Kombinasi terbaik - memanfaatkan kekuatan kedua model
                    
                    Untuk metrik evaluasi yang akurat, unggah model referensi.
                    """)
    
    with gr.Row(variant="panel"):
        gr.Examples(
            examples=[
                "examples/garuda-wisnu-kencana.png",
                "examples/tapel-barong1.png",
                "examples/tapel-barong2.png",
                "examples/pintu-belok.png",
            ],
            inputs=[input_image],
            outputs=[processed_image, output_model_obj, output_model_glb, output_model_ply, f1_metric, uhd_metric, tmd_metric, cd_metric, iou_metric, metrics_text, radar_plot, bar_plot],
            cache_examples=False,
            fn=partial(run_example),
            label="Contoh",
            examples_per_page=20,
        )
    
    submit.click(
        fn=run_generation_pipeline,
        inputs=[
            input_image,
            do_remove_background,
            foreground_ratio,
            mc_resolution,
            reference_model,
            model_quality,
            texture_quality,
            smoothing_factor,
            use_model,
            blend_method,
            model_weight
        ],
        # Kode Baru yang Benar
        outputs=[
            processed_image,
            output_model_obj,
            output_model_glb,
            output_model_ply,   # <-- Tambahkan yang ini
            f1_metric,
            uhd_metric,
            tmd_metric,
            cd_metric,
            iou_metric,
            metrics_text,
            radar_plot,
            bar_plot
        ]
    )
    
    # Dynamic UI updates
    def update_blend_visibility(model_choice):
        if model_choice == "Both":
            return [
                gr.update(visible=True),  # model_weight
                gr.update(visible=True)   # blend_method
            ]
        else:
            return [
                gr.update(visible=False), # model_weight
                gr.update(visible=False)  # blend_method
            ]
    
    use_model.change(
        fn=update_blend_visibility,
        inputs=[use_model],
        outputs=[model_weight, blend_method]
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--username', type=str, default=None, help='Username for authentication')
    parser.add_argument('--password', type=str, default=None, help='Password for authentication')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the server listener on')
    parser.add_argument("--listen", action='store_true', help="launch gradio with 0.0.0.0 as server name")
    parser.add_argument("--share", action='store_true', help="make the UI accessible through gradio.live")
    parser.add_argument("--queuesize", type=int, default=1, help="launch gradio queue max_size")
    
    args = parser.parse_args()
    
    # Configure queue before launch
    interface.queue(max_size=args.queuesize)
    
    # Prepare auth tuple
    auth = None
    if args.username and args.password:
        auth = (args.username, args.password)
    
    # Launch with simplified parameters
    try:
        interface.launch(
            server_port=args.port,
            server_name="0.0.0.0" if args.listen else None,
            share=args.share,
            auth=auth,
            debug=True
        )
    except Exception as e:
        print(f"Failed to launch interface: {str(e)}")
        # Fallback to basic launch if custom configuration fails
        interface.launch()
