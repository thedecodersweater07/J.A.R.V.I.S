using System;
using System.Runtime.InteropServices;

namespace Jarvis.Models
{
    public class JarvisModel : IDisposable
    {
        private bool _disposed = false;
        private IntPtr _modelHandle;

        // Native methods
        [DllImport("JarvisCore")]
        private static extern IntPtr CreateModel(string configPath);

        [DllImport("JarvisCore")]
        private static extern void DeleteModel(IntPtr model);

        [DllImport("JarvisCore")]
        private static extern string ProcessInput(IntPtr model, string input);

        public JarvisModel(string configPath = null)
        {
            _modelHandle = CreateModel(configPath ?? string.Empty);
            if (_modelHandle == IntPtr.Zero)
            {
                throw new Exception("Failed to create JARVIS model");
            }
        }

        public string Process(string input)
        {
            if (string.IsNullOrEmpty(input))
                throw new ArgumentException("Input cannot be null or empty", nameof(input));

            return ProcessInput(_modelHandle, input);
        }

        protected virtual void Dispose(bool disposing)
        {
            if (!_disposed)
            {
                if (disposing)
                {
                    // Free managed resources
                }

                // Free unmanaged resources
                if (_modelHandle != IntPtr.Zero)
                {
                    DeleteModel(_modelHandle);
                    _modelHandle = IntPtr.Zero;
                }

                _disposed = true;
            }
        }

        ~JarvisModel()
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
