package com.example.user.myapplication.ai_group;

import android.graphics.Bitmap;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.user.myapplication.R;

import java.util.ArrayList;
import java.util.List;

public class ai_Adapter extends RecyclerView.Adapter<RecyclerView.ViewHolder>{
    public static final int TYPE_FOOTER = 1;  //说明是带有Footer的
    public static final int TYPE_NORMAL = 2;  //说明是不带有header和footer的

    private List<String> mDatas;
    private View mFooterView;
    private ArrayList<Bitmap> bitmapArray = new ArrayList<Bitmap>();
    private List<String> Friend;
    private List<String> idlist;

    //构造函数
    public ai_Adapter(List<String> list, ArrayList<Bitmap> bitmaps, List<String> friendlist, List<String> idlist){
        this.mDatas = list;
        this.bitmapArray = bitmaps;
        this.Friend = friendlist;
        this.idlist = idlist;
    }

    //FooterView的get和set函数
    public View getFooterView() {
        return mFooterView;
    }
    public void setFooterView(View footerView) {
        mFooterView = footerView;
        notifyItemInserted(getItemCount()-1);
    }

    /** 重写这个方法，很重要，是加入Header和Footer的关键，我们通过判断item的类型，从而绑定不同的view    * */
    @Override
    public int getItemViewType(int position) {
        if (mFooterView == null){
            return TYPE_NORMAL;
        }
        if(position == getItemCount()-1){
            //最后一个,应该加载Footer
            return TYPE_FOOTER;
        }
        return TYPE_NORMAL;
    }

    //创建View，如果是HeaderView或者是FooterView，直接在Holder中返回
    @Override
    public RecyclerView.ViewHolder onCreateViewHolder(ViewGroup parent, int viewType) {
        if(mFooterView != null && viewType == TYPE_FOOTER){
            return new ListHolder(mFooterView);
        }
        View layout = LayoutInflater.from(parent.getContext()).inflate(R.layout.ai_group_item, parent, false);

        return new ListHolder(layout);
    }

    //绑定View，这里是根据返回的这个position的类型，从而进行绑定的，   HeaderView和FooterView, 就不同绑定了
    @Override
    public void onBindViewHolder(RecyclerView.ViewHolder holder, final int position) {
        if(getItemViewType(position) == TYPE_NORMAL){
            if(holder instanceof ListHolder) {
                //这里加载数据的时候要注意，是从position-1开始，因为position==0已经被header占用了
                ((ListHolder) holder).tv.setText(mDatas.get(position));
                ((ListHolder) holder).iw.setImageBitmap(bitmapArray.get(position));
                for (int a = 0; a < idlist.size(); a++){
                    if (idlist.get(a).equals(Friend.get(position))){
                        ((ListHolder) holder).checkBox.setChecked(true);
                    }
                }

                ((ListHolder) holder).checkBox.setOnClickListener(new View.OnClickListener() {
                    @Override
                    public void onClick(View v) {
                        if (((CheckBox) v).isChecked()){
                            idlist.add(Friend.get(position));
                            Log.d("check",":"+idlist);
                        }else {
                            for (int i = 0; i < idlist.size(); i++){
                                if (idlist.get(i).equals(Friend.get(position))){
                                    idlist.remove(i);
                                }
                            }
                            Log.d("re",":"+idlist);
                        }
                    }
                });
                return;
            }
            return;
        }else {
            return;
        }
    }

    //在这里面加载ListView中的每个item的布局
    class ListHolder extends RecyclerView.ViewHolder{
        TextView tv;
        ImageView iw;
        CheckBox checkBox;
        public ListHolder(View itemView) {
            super(itemView);
            //如果是footerview,直接返回
            if (itemView == mFooterView){
                return;
            }
            tv = itemView.findViewById(R.id.name);
            iw = itemView.findViewById(R.id.photo);
            checkBox = itemView.findViewById(R.id.addfriend_checkbox);
        }
    }

    //返回View中Item的个数，这个时候，总的个数应该是ListView中Item的个数加上HeaderView和FooterView
    @Override
    public int getItemCount() {
        if(mFooterView == null){
            return mDatas.size();
        }else{
            return mDatas.size() + 1;
        }
    }

}
