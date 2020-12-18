// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) SignalPop LLC. All rights reserved.
// Licensed under the MIT License.

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Microsoft.ML.OnnxTraining
{
    /// <summary>
    /// Holds the training parameters used by the TrainingSession.
    /// </summary>
    public class TrainingParameters : IDisposable
    {
        /// <summary>
        /// A pointer to a underlying native instance of OrtTrainingParameters
        /// </summary>
        protected IntPtr _nativeHandle;
        private OrtDataGetBatchCallback m_fnGetTrainingData;
        private OrtDataGetBatchCallback m_fnGetTestingData;
        private OrtErrorFunctionCallback m_fnErrorFunction;
        private OrtEvaluationFunctionCallback m_fnEvaluateFunction;
        public event EventHandler<ErrorFunctionArgs> OnErrorFunction;
        public event EventHandler<EvaluationFunctionArgs> OnEvaluationFunction;
        public event EventHandler<DataBatchArgs> OnGetTrainingDataBatch;
        public event EventHandler<DataBatchArgs> OnGetTestingDataBatch;
        private DisposableList<IDisposable> m_rgCleanUpList = new DisposableList<IDisposable>();
        private List<KeyValuePair<string, List<int>>> m_rgExpectedInputs;
        private List<KeyValuePair<string, List<int>>> m_rgExpectedOutputs;
        private bool _disposed = false;

        /// <summary>
        /// Constructs a default TrainingParameters
        /// </summary>
        public TrainingParameters()
        {
            m_fnGetTrainingData = getTrainingDataFn;
            m_fnGetTestingData = getTestingDataFn;
            m_fnErrorFunction = errorFn;
            m_fnEvaluateFunction = evaluationFn;
            Init();
        }

        private void Init()
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtCreateTrainingParameters(out _nativeHandle));
        }

        /// <summary>
        /// Finalizer. to cleanup training parameters in case it runs
        /// and the user forgets to Dispose() of the training parameters.
        /// </summary>
        ~TrainingParameters()
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
                m_rgCleanUpList.Dispose();
            }

            // cleanup unmanaged resources
            if (_nativeHandle != IntPtr.Zero)
            {
                NativeMethodsTraining.OrtReleaseTrainingParameters(_nativeHandle);
                _nativeHandle = IntPtr.Zero;
            }
            _disposed = true;
        }

        #endregion


        #region Public Properties

        public IntPtr DangerousGetHandle()
        {
            return _nativeHandle;
        }

        public List<KeyValuePair<string, List<int>>> ExpectedInputs
        {
            get { return m_rgExpectedInputs; }
            set { m_rgExpectedInputs = value; }
        }

        public List<KeyValuePair<string, List<int>>> ExpectedOutputs
        {
            get { return m_rgExpectedOutputs; }
            set { m_rgExpectedOutputs = value; }
        }

        public void SetTrainingParameter(OrtTrainingStringParameter key, string strVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_string(_nativeHandle, key, NativeMethods.GetPlatformSerializedString(strVal)));
        }

        public string GetTrainingParameter(OrtTrainingStringParameter key)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_string(_nativeHandle, key, allocator.Pointer, out valHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(valHandle);
            }
            return str;
        }

        public void SetTrainingParameter(OrtTrainingBooleanParameter key, bool bVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_bool(_nativeHandle, key, bVal));
        }

        public bool GetTrainingParameter(OrtTrainingBooleanParameter key)
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_bool(_nativeHandle, key, out val));

            if ((ulong)val == 0)
                return false;
            else
                return true;
        }

        public void SetTrainingParameter(OrtTrainingLongParameter key, long lVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetParameter_long(_nativeHandle, key, lVal));
        }

        public long GetTrainingParameter(OrtTrainingLongParameter key)
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetParameter_long(_nativeHandle, key, out val));

            return (long)val;
        }

        public void SetTrainingParameter(OrtTrainingNumericParameter key, double dfVal)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetNumericParameter(_nativeHandle, key, dfVal));
        }

        public double GetTrainingParameter(OrtTrainingNumericParameter key)
        {
            string str = null;
            var allocator = OrtAllocator.DefaultInstance;
            IntPtr valHandle = IntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetNumericParameter(_nativeHandle, key, allocator.Pointer, out valHandle));

            using (var ortAllocation = new OrtMemoryAllocation(allocator, valHandle, 0))
            {
                str = NativeOnnxValueHelper.StringFromNativeUtf8(valHandle);
            }
            return double.Parse(str);
        }

        public void SetTrainingOptimizer(OrtTrainingOptimizer opt)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetTrainingOptimizer(_nativeHandle, opt));
        }

        public OrtTrainingOptimizer GetTrainingOptimizer()
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetTrainingOptimizer(_nativeHandle, out val));

            switch (((OrtTrainingOptimizer)(int)val))
            {
                case OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD:
                    return OrtTrainingOptimizer.ORT_TRAINING_OPTIMIZER_SGD;

                default:
                    throw new Exception("Unknown optimizer '" + val.ToString() + "'!");
            }
        }

        public void SetTrainingLossFunction(OrtTrainingLossFunction loss)
        {
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetTrainingLossFunction(_nativeHandle, loss));
        }

        public OrtTrainingLossFunction GetTrainingLossFunction()
        {
            UIntPtr val = UIntPtr.Zero;
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtGetTrainingLossFunction(_nativeHandle, out val));

            switch (((OrtTrainingLossFunction)(int)val))
            {
                case OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY:
                    return OrtTrainingLossFunction.ORT_TRAINING_LOSS_FUNCTION_SOFTMAXCROSSENTROPY;

                default:
                    throw new Exception("Unknown loss function '" + val.ToString() + "'!");
            }
        }

        #endregion

        #region Public Methods

        public void errorFn(IntPtr colVal)
        {
            if (OnErrorFunction == null)
                return;

            List<DisposableNamedOnnxValue> rgVal = new List<DisposableNamedOnnxValue>();
            List<string> rgNames = new List<string>();
            OrtValueCollection col = new OrtValueCollection(colVal);

            int nCount = col.Count;
            for (int i = 0; i < nCount; i++)
            {
                string strName;
                OrtValue val = col.GetAt(i, out strName);
                rgVal.Add(DisposableNamedOnnxValue.CreateTensorFromOnnxValue(strName, val));
            }

            OnErrorFunction(this, new ErrorFunctionArgs(rgVal));

            // Clean-up the data used during this batch.
            foreach (IDisposable iDispose in m_rgCleanUpList)
            {
                iDispose.Dispose();
            }

            m_rgCleanUpList.Clear();
        }

        public void evaluationFn(long lNumSamples, long lStep)
        {
            if (OnEvaluationFunction == null)
                return;

            OnEvaluationFunction(this, new EvaluationFunctionArgs(lNumSamples, lStep));
        }

        public void SetupTrainingParameters()
        {
            Guid guid = System.Guid.NewGuid();
            string strKey = guid.ToString();
            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetupTrainingParameters(_nativeHandle, m_fnErrorFunction, m_fnEvaluateFunction, NativeMethods.GetPlatformSerializedString(strKey)));
        }

        public void getTrainingDataFn(long nBatchSize, IntPtr hVal)
        {
            if (OnGetTestingDataBatch == null)
                return;

            DataBatchArgs args = new DataBatchArgs(nBatchSize, m_rgExpectedInputs, m_rgExpectedOutputs);
            OnGetTrainingDataBatch(this, args);
            handleGetDataFn(args, hVal);
        }

        public void getTestingDataFn(long nBatchSize, IntPtr hVal)
        {
            if (OnGetTestingDataBatch == null)
                return;

            DataBatchArgs args = new DataBatchArgs(nBatchSize, m_rgExpectedInputs, m_rgExpectedOutputs);
            OnGetTestingDataBatch(this, args);
            handleGetDataFn(args, hVal);
        }

        private void handleGetDataFn(DataBatchArgs args, IntPtr hcol)
        {
            OrtValueCollection col = new OrtValueCollection(hcol);

            for (int i = 0; i < args.Values.Count; i++)
            {
                MemoryHandle? memHandle;
                OrtValue val = args.Values[i].ToOrtValue(out memHandle);

                if (memHandle.HasValue)
                    m_rgCleanUpList.Add(memHandle);

                m_rgCleanUpList.Add(val);
              
                col.SetAt(i, val, args.Values[i].Name);
            }
        }

        public void SetupTrainingData(List<string> rgstrFeedNames)
        {
            string strFeedNames = "";
            
            for (int i = 0; i < rgstrFeedNames.Count; i++)
            {
                strFeedNames += rgstrFeedNames[i];
                strFeedNames += ";";
            }

            NativeApiStatus.VerifySuccess(NativeMethodsTraining.OrtSetupTrainingData(_nativeHandle, m_fnGetTrainingData, m_fnGetTestingData, NativeMethods.GetPlatformSerializedString(strFeedNames)));
        }

        #endregion
    }

    public class ErrorFunctionArgs : EventArgs
    {
        List<DisposableNamedOnnxValue> m_rgVal;

        public ErrorFunctionArgs(List<DisposableNamedOnnxValue> rgVal)
        {
            m_rgVal = rgVal;
        }

        public List<DisposableNamedOnnxValue> Values
        {
            get { return m_rgVal; }
        }

        public DisposableNamedOnnxValue Find(string strName)
        {
            foreach (DisposableNamedOnnxValue val in m_rgVal)
            {
                if (val.Name == strName)
                    return val;
            }

            return null;
        }
    }

    public class EvaluationFunctionArgs : EventArgs 
    {
        long m_lNumSamples;
        long m_lStep;

        public EvaluationFunctionArgs(long lNumSamples, long lStep)
        {
            m_lNumSamples = lNumSamples;
            m_lStep = lStep;
        }

        public long NumSamples
        {
            get { return m_lNumSamples; }
        }

        public long Step
        {
            get { return m_lStep; }
        }
    }

    public class DataBatchArgs : EventArgs
    {
        int m_nBatchSize;
        List<NamedOnnxValue> m_rgValues = new List<NamedOnnxValue>();
        List<KeyValuePair<string, List<int>>> m_rgExpectedInputs;
        List<KeyValuePair<string, List<int>>> m_rgExpectedOutputs;

        public DataBatchArgs(long nBatchSize, List<KeyValuePair<string, List<int>>> rgExpectedInputs, List<KeyValuePair<string, List<int>>> rgExpectedOutputs)
        {
            m_nBatchSize = (int)nBatchSize;
            m_rgExpectedInputs = rgExpectedInputs;
            m_rgExpectedOutputs = rgExpectedOutputs;
        }

        public List<NamedOnnxValue> Values
        {
            get { return m_rgValues; }
        }

        public int BatchSize
        {
            get { return m_nBatchSize; }
        }

        public List<KeyValuePair<string, List<int>>> Inputs
        {
            get { return m_rgExpectedInputs; }
        }

        public List<KeyValuePair<string, List<int>>> Outpus
        {
            get { return m_rgExpectedOutputs; }
        }

        public List<int> GetInputAt(int nIdx, out string strName)
        {
            List<int> rg = new List<int>();

            for (int i = 1; i < m_rgExpectedInputs[nIdx].Value.Count; i++)
            {
                rg.Add(m_rgExpectedInputs[nIdx].Value[i]);
            }

            strName = m_rgExpectedInputs[nIdx].Key;

            while (rg.Count > 0 && rg[rg.Count - 1] == 1)
            {
                rg.RemoveAt(rg.Count - 1);
            }

            return rg;
        }

        public List<int> GetOutputAt(int nIdx, out string strName)
        {
            List<int> rg = new List<int>();

            for (int i = 1; i < m_rgExpectedOutputs[nIdx].Value.Count; i++)
            {
                rg.Add(m_rgExpectedOutputs[nIdx].Value[i]);
            }

            strName = m_rgExpectedOutputs[nIdx].Key;

            while (rg.Count > 0 && rg[rg.Count - 1] == 1)
            {
                rg.RemoveAt(rg.Count - 1);
            }

            return rg;
        }
    }
}
