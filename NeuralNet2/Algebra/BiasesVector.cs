using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public class BiasesVector : NetMatrix
    {
        #region constructors
        public BiasesVector(MatrixIdentifier id)
            : base (id)
        { }
        #endregion


        #region properties
        public int Dimension { get { return AlgebraManager.GetColumnCount(MatrixID); } }
        #endregion
    }
}
