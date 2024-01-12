import numpy as np
import matplotlib.pyplot as plt
import os
import pywt
from tqdm import tqdm

def execute_all_statistical_analysis(data_dict, config):
    # Get the current working directory
    cwd = os.getcwd()
    for task_type, task_data in data_dict.items():
        for model_name, model_data in task_data.items():
            for video_name, data in model_data.items():
                if task_type == "depth_estimation":
                    print(video_name)
                    frames = data
                    stacked_frames = np.stack([frame[0, 0] for frame in frames])

                    # Define the directory to save the plot for each video
                    save_dir = os.path.join(cwd, 'results', 'statistics', 'fft', video_name)

                    # Create the directory if it doesn't exist
                    os.makedirs(save_dir, exist_ok=True)

                    # Now, stacked_frames is a 3D array of shape (num_frames, 720, 1280)
                    # Apply FFT along the time dimension (axis=0)
                    fft_result = np.fft.fft(stacked_frames, axis=0)

                    # To analyze the frequencies, look at the magnitude of the FFT
                    # Save the FFT result for each video
                    # np.save(os.path.join(save_dir, 'fft_result.npy'), fft_result)

                    # To analyze the frequencies, look at the magnitude of the FFT
                    fft_magnitude = np.abs(fft_result)

                    # You can now plot or analyze fft_magnitude to find patterns over time
                    # For example, plotting the FFT magnitude for a specific pixel
                    average_spectrum = np.mean(fft_magnitude, axis=(1, 2))

                    # Plot the average spectrum
                    plt.plot(average_spectrum)
                    plt.title("Average Frequency Spectrum")
                    plt.xlabel("Frequency")
                    plt.ylabel("Magnitude")

                    # Save the plot in the directory for the current video
                    plt.savefig(os.path.join(save_dir, 'average_spectrum.png'))
                    plt.clf()

                    # Perform a Continuous Wavelet Transform (CWT) with a Ricker (or "Mexican hat") wavelet
                    wavelet = 'morl'
                    scales = np.logspace(0, 2, num=20)

                    # Initialize an array to store the wavelet transform for all frames
                    all_coefficients = np.zeros((len(stacked_frames), len(scales), stacked_frames.shape[1], stacked_frames.shape[2]))

                    for i in tqdm(range(len(stacked_frames))):
                        coefficients, frequencies = pywt.cwt(stacked_frames[i], scales, wavelet)
                        all_coefficients[i] = coefficients


                    # Calculate the average wavelet transform across all frames
                    average_coefficients = np.mean(all_coefficients, axis=0)
                    # Loop over the scales
                    for j in range(len(scales)):
                        print(average_coefficients[j].shape)
                        # Plot the average wavelet transform for the current scale
                        plt.imshow(abs(average_coefficients[j]), aspect='auto', cmap='hot', 
                                extent=[0, len(stacked_frames), frequencies.min(), frequencies.max()])
                        plt.colorbar(label="Magnitude")
                        plt.title(f'Average Time-Frequency Representation of Signal (Scale {j})')
                        plt.xlabel('Time')
                        plt.ylabel('Frequency')
                    
                        # Save the plot in the directory for the current video
                        plt.savefig(os.path.join(save_dir, f'average_wavelet_transform_scale_{j}.png'))
                        plt.clf()

    return print('done')