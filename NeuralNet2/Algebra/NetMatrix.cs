using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet.Algebra
{
    public abstract class NetMatrix
    {
        public MatrixIdentifier MatrixID { get; private set; }

        public NetMatrix(MatrixIdentifier id)
        {
            MatrixID = id;
        }
    }
}
