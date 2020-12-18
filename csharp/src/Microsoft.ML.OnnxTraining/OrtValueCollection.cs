// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// The OrtValueCollection holds ort values returned by the error function, but does not actually own any of them
    /// and therefore does not release them.
    /// </summary>
    internal class OrtValueCollection : IDisposable
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtValueCollection
        /// </summary>
        protected IntPtr _nativeHandle;
        /// <summary>
        /// Specifies whether or not the native handle is owned.
        /// </summary>
        protected bool _ownsHandle;
        private bool _disposed = false;


        /// <summary>
        /// The OrtValueCollection is an object is not a native collection, but instead
        /// gives access to a group of native OrtValues via its GetAt and SetAt methods.
        /// </summary>
        /// <param name="h">Specifies the handle to the native OrtValueCollection to use, or IntPtr.Zero.  
        /// If IntPtr.Zero, the OrtValueCollection creates a value collection that it owns and disposes,
        /// otherwise the OrtValueCollection does not own the collection and therefore does not dispose it.</param>
        /// <remarks>
        /// For efficiency, the OrtValue collection gives access to a set of OrtValues where
        /// each OrtValue does not actually own the memory but instead points to one or 
        /// more pre-allocated OrtValues. 
        /// </remarks>
        public OrtValueCollection(IntPtr h)
        {
            if (h == IntPtr.Zero)
            {
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtCreateValueCollection(out _nativeHandle));
                _ownsHandle = true;
            }
            else
            {
                _nativeHandle = h;
                _ownsHandle = false;
            }
        }

        /// <summary>
        /// Finalizer. to cleanup training parameters in case it runs
        /// and the user forgets to Dispose() of the training parameters.
        /// </summary>
        ~OrtValueCollection()
        {
            Dispose(false);
        }

        #region Disposable

        /// <summary>
        /// Release all resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            // Suppress finalization.
            GC.SuppressFinalize(this);
        }

        /// <summary>
        /// IDisposable implementation
        /// </summary>
        /// <param name="disposing">true if invoked from Dispose() method</param>
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            // dispose managed state (managed objects).
            if (disposing)
            {
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                if (_ownsHandle)
                    NativeMethodsTraining.OrtReleaseValueCollection(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion


        #region Public Methods

        public bool IsOwned
        {
            get { return _ownsHandle; }
        }

        public IntPtr DangerousGetHandle()
        {
            return _nativeHandle;
        }

        public int Count
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCount(_nativeHandle, out val));
                return (int)val;
            }
        }

        public int Capacity
        {
            get
            {
                UIntPtr val = UIntPtr.Zero;
                NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetCapacity(_nativeHandle, out val));
                return (int)val;
            }
        }

        public OrtValue GetAt(int nIdx, out string strName)
        {
            IntPtr valData;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valName;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetAt(_nativeHandle, nIdx, out valData, allocator.Pointer, out valName));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valName, 0))
            {
                strName = NativeOnnxValueHelper.StringFromNativeUtf8(valName);
            }

            return new OrtValue(valData, false);
        }

        public void SetAt(int nIdx, OrtValue val, string strName = "")
        {
            byte[] rgName = (string.IsNullOrEmpty(strName)) ? null : NativeMethods.GetPlatformSerializedString(strName);
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetAt(_nativeHandle, nIdx, val.Handle, rgName));
        }

        #endregion
    }
}
