using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public class WeightsMatrix : NetMatrix
    {
        #region constructors
        public WeightsMatrix(MatrixIdentifier id)
            : base (id)
        { }
        #endregion

        #region properties
        public int NumberOfInputs { get { return AlgebraManager.GetColumnCount(MatrixID); } }
        public int NumberOfOutputs { get { return AlgebraManager.GetRowCount(MatrixID); } }
        #endregion
    }
}
