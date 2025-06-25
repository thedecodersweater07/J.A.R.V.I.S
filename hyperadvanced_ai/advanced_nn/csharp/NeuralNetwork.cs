using System;
using System.Runtime.InteropServices;
using System.Collections.Generic;

namespace Jarvis.HyperAI.AdvancedNN
{
    /// <summary>
    /// High-performance neural network implementation with native interop
    /// </summary>
    public class NeuralNetwork : IDisposable
    {
        private IntPtr _nativeHandle;
        private bool _disposed = false;

        // Native methods
        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr CreateNeuralNetwork();

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern void DeleteNeuralNetwork(IntPtr handle);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool InitializeNeuralNetwork(IntPtr handle, string configPath);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern IntPtr Predict(IntPtr handle, float[] input, int inputSize, out int outputSize);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        private static extern void Train(IntPtr handle, float[][] inputs, int inputCount, int inputSize,
                                        float[][] targets, int targetSize, int epochs);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool SaveWeights(IntPtr handle, string path);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool LoadWeights(IntPtr handle, string path);

        [DllImport("hyperai_native", CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        private static extern bool IsInitialized(IntPtr handle);

        /// <summary>
        /// Create a new NeuralNetwork instance
        /// </summary>
        public NeuralNetwork()
        {
            _nativeHandle = CreateNeuralNetwork();
            if (_nativeHandle == IntPtr.Zero)
                throw new Exception("Failed to create native NeuralNetwork instance");
        }

        /// <summary>
        /// Initialize the neural network with a configuration file
        /// </summary>
        /// <param name="configPath">Path to the configuration file</param>
        /// <returns>True if initialization was successful</returns>
        public bool Initialize(string configPath = null)
        {
            if (_disposed)
                throw new ObjectDisposedException("NeuralNetwork");

            return InitializeNeuralNetwork(_nativeHandle, configPath ?? string.Empty);
        }

        /// <summary>
        /// Perform inference on input data
        /// </summary>
        /// <param name="input">Input data</param>
        /// <returns>Model predictions</returns>
        public float[] Predict(float[] input)
        {
            if (_disposed)
                throw new ObjectDisposedException("NeuralNetwork");

            if (input == null || input.Length == 0)
                throw new ArgumentException("Input cannot be null or empty", nameof(input));

            IntPtr resultPtr = Predict(_nativeHandle, input, input.Length, out int outputSize);
            if (resultPtr == IntPtr.Zero || outputSize == 0)
                return Array.Empty<float>();

            float[] result = new float[outputSize];
            Marshal.Copy(resultPtr, result, 0, outputSize);
            
            // Free the unmanaged memory allocated in the native code
            Marshal.FreeCoTaskMem(resultPtr);
            
            return result;
        }

        /// <summary>
        /// Train the neural network
        /// </summary>
        /// <param name="inputs">Training data</param>
        /// <param name="targets">Target values</param>
        /// <param name="epochs">Number of training epochs</param>
        public void Train(float[][] inputs, float[][] targets, int epochs = 1)
        {
            if (_disposed)
                throw new ObjectDisposedException("NeuralNetwork");

            if (inputs == null || inputs.Length == 0)
                throw new ArgumentException("Inputs cannot be null or empty", nameof(inputs));
            
            if (targets == null || targets.Length == 0)
                throw new ArgumentException("Targets cannot be null or empty", nameof(targets));
            
            if (inputs.Length != targets.Length)
                throw new ArgumentException("Inputs and targets must have the same length");
                
            if (epochs < 1)
                throw new ArgumentOutOfRangeException(nameof(epochs), "Number of epochs must be at least 1");

            Train(_nativeHandle, inputs, inputs.Length, inputs[0].Length, 
                 targets, targets[0].Length, epochs);
        }

        /// <summary>
        /// Save the model weights to a file
        /// </summary>
        /// <param name="path">Path to save the weights</param>
        /// <returns>True if successful</returns>
        public bool SaveWeights(string path)
        {
            if (_disposed)
                throw new ObjectDisposedException("NeuralNetwork");

            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path cannot be null or whitespace", nameof(path));

            return SaveWeights(_nativeHandle, path);
        }

        /// <summary>
        /// Load model weights from a file
        /// </summary>
        /// <param name="path">Path to the weights file</param>
        /// <returns>True if successful</returns>
        public bool LoadWeights(string path)
        {
            if (_disposed)
                throw new ObjectDisposedException("NeuralNetwork");

            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("Path cannot be null or whitespace", nameof(path));

            if (!System.IO.File.Exists(path))
                throw new System.IO.FileNotFoundException("Weights file not found", path);

            return LoadWeights(_nativeHandle, path);
        }

        /// <summary>
        /// Check if the network is initialized
        /// </summary>
        public bool IsInitialized
        {
            get
            {
                if (_disposed)
                    throw new ObjectDisposedException("NeuralNetwork");
                    
                return IsInitialized(_nativeHandle);
            }
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (_nativeHandle != IntPtr.Zero)
                {
                    DeleteNeuralNetwork(_nativeHandle);
                    _nativeHandle = IntPtr.Zero;
                }
                _disposed = true;
            }
        }

        ~NeuralNetwork()
        {
            Dispose(false);
        }

        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }
    }
}
