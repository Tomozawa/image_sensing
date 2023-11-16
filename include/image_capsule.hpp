//処理をコピーではなくムーブでするためのMatのカプセル
//Opnecvの処理関数が(入力された画像に対して加工するのではなく)コピー前提で書かれているため、処理関数を全部書き直す必要あり。むり

#include <opencv2/core/mat.hpp>

namespace image_capsule{
    using namespace cv;

    template<typename T>
    concept State = requires{
        typename T::NextState;
    };

    template<State T>
    class SiloImage final{
    private:
        Mat mat;
    public:
        SiloImage(SiloImage<T>&& silo_img): mat(silo_img.mat){
            silo_img.mat.release();
        }

        SiloImage(const SiloImage&) = delete;

        SiloImage(Mat&& mat): mat(mat){
            //Matはボクシングされたオブジェクトへのスマートポインタ
            //Mat自体が画像への参照なので、Matは普通にコピーしてしまう
            mat.release();
        }

        SiloImage<T> operator=(const SiloImage<T>&) = delete;

        SiloImage<T>& operator=(SiloImage<T>&& silo_img){
            mat = silo_img.mat;
            silo_img.mat.release();
        }

        Mat to_mat(){
            //Mat自体が画像への参照なので、Matは普通にコピーしてしまう
            Mat result = mat;
            mat.release();
            return result;
        }

        ~SiloImage() = default;
    };
} //namespace image_capsule

namespace image_capsule::states{
    struct SILO_RAW;
    struct SILO_HSV_CONVERTED;
    struct SILO_BLURED;
    struct SILO_MASKED;
    struct SILO_EDGE_EXTRACTED;
    struct STATE_NONE{using NextState = STATE_NONE;};

    struct SILO_RAW{
        using NextState = SILO_HSV_CONVERTED;
    };

    struct SILO_HSV_CONVERTED{
        using NextState = SILO_BLURED;
    };

    struct SILO_BLURED{
        using NextState = SILO_MASKED;
    };

    struct SILO_MASKED{
        using NextState = SILO_EDGE_EXTRACTED;
    };

    struct SILO_EDGE_EXTRACTED{
        using NextState = STATE_NONE;
    };
}//imagecapsule::states

using namespace image_capsule;