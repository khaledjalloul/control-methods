template <typename T>
T LTIStateSpaceModel::y(T x)
{
    return C * x;
}