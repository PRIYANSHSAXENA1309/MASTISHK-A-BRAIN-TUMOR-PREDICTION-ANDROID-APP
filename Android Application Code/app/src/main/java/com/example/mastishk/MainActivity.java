package com.example.mastishk;

import android.animation.Animator;
import android.animation.AnimatorListenerAdapter;
import android.content.Context;
import android.content.Intent;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.FrameLayout;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import com.airbnb.lottie.LottieAnimationView;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    ImageView img;
    Button btnselect, btnpredict;
    TextView txtoutput;
    Bitmap images, image;
    LottieAnimationView lottieanim;
    FrameLayout frame;
    private Interpreter tflite;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);
        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        try {
            tflite = new Interpreter(loadModelFile(this));
        } catch (IOException e) {
            Log.e("Tflite", "Error loading model", e);
        }

        img = findViewById(R.id.img_brain);
        btnselect = findViewById(R.id.btn_select);
        btnpredict = findViewById(R.id.btn_predict);
        txtoutput = findViewById(R.id.txt_output);
        frame = findViewById(R.id.overlay);
        lottieanim = findViewById(R.id.anim2);

        btnselect.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
                intent.setType("image/*");
                startActivityForResult(intent,500);
            }
        });

        btnpredict.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                if(images != null) {
                    frame.setVisibility(View.VISIBLE);
                    frame.bringToFront();
                    lottieanim.playAnimation();
                    lottieanim.addAnimatorListener(new AnimatorListenerAdapter() {
                        @Override
                        public void onAnimationEnd(Animator animation) {
                            super.onAnimationEnd(animation);
                            frame.setVisibility(View.GONE);
                            Predict();
                        }
                    });
                } else {
                    Toast.makeText(getApplicationContext(),"No Image is Selected",Toast.LENGTH_SHORT).show();
                }
            }
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if(requestCode == 500) {
            img.setImageURI(data.getData());
            Uri uri = data.getData();
            try {
                images = MediaStore.Images.Media.getBitmap(this.getContentResolver(), uri);
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private void Predict(){
        image = Bitmap.createScaledBitmap(images, 64, 64, true);
        try {
            TensorImage tensorImage = new TensorImage(DataType.FLOAT32);
            tensorImage.load(image);
            TensorBuffer inputFeature = TensorBuffer.createFixedSize(new int[]{1,64,64,3}, DataType.FLOAT32);
            inputFeature.loadBuffer(tensorImage.getBuffer());
            TensorBuffer outputFeature = TensorBuffer.createFixedSize(new int[]{1,1}, DataType.FLOAT32);
            tflite.run(inputFeature.getBuffer(),outputFeature.getBuffer());
            float result = outputFeature.getFloatArray()[0];
            if(result == 1.0){
                txtoutput.setText("Yes Tumor detected");
            }
            else if (result == 0.0){
                txtoutput.setText("No tumor detected");
            }
        } catch (Exception e) {
            Log.e("Tflite","Error during prediction", e);
            txtoutput.setText("Error in Prediction");
        }
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd("model.tflite");
        FileInputStream inputStream = fileDescriptor.createInputStream();
        FileChannel filechannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return filechannel.map(FileChannel.MapMode.READ_ONLY,startOffset,declaredLength);
    }
    @Override
    protected void onDestroy(){
        super.onDestroy();
        if(tflite != null){
            tflite.close();
        }
    }
}