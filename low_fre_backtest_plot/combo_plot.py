# -*- encoding: utf-8 -*-
'''
@File    :   plot.py
@Time    :   2021/12/28 13:47:42
@Author  :   Wenqian (Bradley) He
@Version :   1.3
@Contact :   vincent_wq@outlook.com
'''
# 将数据重组，画图，拆分成独立的类，再用模板决定布局，增加扩展性
# 基于生成器设计模式设计
# here put the import lib

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


class Template:
    '''
        不同布局模板的具体实现，今后有不同的需求可以添加模板
    '''

    def __init__(self, fig_width: int, fig_height: int, sample_interval: int) -> None:
        self.fig_width = fig_width
        self.fig_height = fig_height
        self.sample_interval = sample_interval

    class Chart:
        '''
            决定画图类型的类
        '''

        def __init__(self, fig_width: int, fig_height: int, sample_interval: int) -> None:
            self.fig_width = fig_width
            self.fig_height = fig_height
            self.interval = sample_interval

        @staticmethod
        def color_collection_judge(df: pd.DataFrame, color_collection: object) -> list:
            '''
                可以按照color dict或者color list定制和数据字段绑定的映射
                :param df - 数据dataframe
                :param color_collection - 颜色映射
            '''
            if color_collection:
                if isinstance(color_collection, dict):
                    colors = [color_collection.get(x, '#333333') for x in df.columns]
                elif isinstance(color_collection, list):
                    colors = color_collection
            else:
                colors = None
            return colors

        @staticmethod
        def style_collection_judge(df: pd.DataFrame, style_collection: object) -> list:
            '''
                可以按照style dict或者style list定制和数据字段绑定的映射
                :param df - 数据dataframe
                :param style_collection - 颜色映射
            '''
            if style_collection:
                if isinstance(style_collection, dict):
                    styles = [style_collection.get(x, '#333333') for x in df.columns]
                elif isinstance(style_collection, list):
                    styles = style_collection
            else:
                styles = None
            return styles

        @staticmethod
        def index_type_judge(df: pd.DataFrame) -> pd.DataFrame:
            '''
                确定dataframe的index是string
                :param df - 数据dataframe
            '''
            df.index = [str(index) for index in df.index.tolist()]
            return df

        def line_chart(
                self, df: pd.DataFrame, ax: object, title: str, color_collection=None,
                style_collection=None
        ) -> None:
            '''
                折线图抽象类型
            '''
            df = self.index_type_judge(df=df)
            colors = self.color_collection_judge(df=df, color_collection=color_collection)
            styles = self.style_collection_judge(df=df, style_collection=style_collection)
            # plt.plot
            cols = df.columns
            x = np.arange(len(df))
            cnt = 0
            if type(color_collection) == type({}):
                color_collection = [color_collection[col] for col in cols]
            if type(style_collection) == type({}):
                style_collection = [style_collection[col] for col in cols]
            for col in cols:
                if (not color_collection is None) and (not style_collection is None):
                    plt.plot(x, df[col], color=color_collection[cnt], linestyle=style_collection[cnt], linewidth=2.5,
                             label=col)
                elif (not color_collection is None):
                    plt.plot(x, df[col], color=color_collection[cnt], linewidth=2.5, label=col)
                elif (not style_collection is None):
                    plt.plot(x, df[col], linestyle=style_collection[cnt], linewidth=2.5, label=col)
                else:
                    plt.plot(x, df[col], linewidth=2.5, label=col)
                cnt += 1
            # df.plot(kind='line', ax=ax, color=colors, style=styles, linewidth=2.4)
            # 设置图例
            ax.legend(fontsize=self.fig_width / 5, loc='upper left')
            # 设置x轴tick与label
            x_labels = list(df.index)
            x_labels_pos = range(0, len(x_labels), self.interval)
            ax.set_xticks(x_labels_pos)
            x_labels = [x for x in x_labels if x_labels.index(x) in x_labels_pos]
            ax.set_xticklabels(x_labels, fontsize=self.fig_width // 4.5)#, rotation=10, ha='right')
            # 设置y轴tick与label
            plt.yticks(fontsize=self.fig_width // 4)
            plt.title(title, fontsize=self.fig_width // 4)
            plt.grid()

        def bar_chart(
                self, df: pd.DataFrame, ax: object, title: str, y_label_left: str, y_label_right: str,
                stacked=False, color_collection=None
        ) -> None:
            '''
                柱状图抽象类型
            '''
            df = self.index_type_judge(df=df)
            # print(df)
            max_value_left = df.iloc[:, 0].abs().max() * 1.1
            max_value_right = df.iloc[:, 1].abs().max() * 1.1
            colors = self.color_collection_judge(df=df, color_collection=color_collection)
            # plt.bar(df)
            cols = df.columns
            x = np.arange(len(df))
            # i = 0
            # cnt = 0
            # for col in cols:
            #     if (not color_collection is None):
            #         plt.bar(x + i, df[col], color=color_collection[cnt], width=0.3, label=col, alpha=0.7)
            #     else:
            #         plt.bar(x + i, df[col], width=0.3, label=col, alpha=0.7)
            #     i += 0.3
            #     cnt += 1
            # df.plot(
            #     kind='bar', ax=ax, stacked=stacked,
            #     color=colors
            # )
            plt.axis()
            labels = list(df.index)
            ax.set_xticks(np.arange(len(labels)) + 0.3)
            ax.set_xticklabels(labels, fontsize=self.fig_width // 4, rotation=10, ha='right')
            if len(df) >= 13:
                ax.set_xticklabels(labels, fontsize=self.fig_width / 5, rotation=10, ha='right')
            # 设置y轴tick与label
            plt.yticks(fontsize=self.fig_width // 4)
            if 'beta' in labels:
                ax.set_ylim(min(-1, -max_value_left), max(1, max_value_left))
            else:
                ax.set_ylim(-max_value_left, max_value_left)
            ax.set_ylabel(y_label_left, fontsize=self.fig_width // 4)
            if y_label_right:
                ax1 = ax.twinx()
                ax1.set_ylabel(y_label_right, fontsize=self.fig_width // 4)
                plt.yticks(fontsize=self.fig_width // 4)
                ax1.set_ylim(-max_value_right, max_value_right)

            ax.bar(x, df.iloc[:, 0], color=color_collection[0], width=0.3, label=cols[0], alpha=0.7)
            ax1.bar(x + 0.3, df.iloc[:, 1], color=color_collection[1], width=0.3, label=cols[1], alpha=0.7)
            ax.legend(fontsize=self.fig_width // 4.5, loc='upper right')
            ax1.legend(fontsize=self.fig_width // 4.5, loc='upper left')

            plt.title(title, fontsize=self.fig_width // 4)
            plt.grid()

        def bar_ts_chart(
            self, df: pd.DataFrame, ax: object, title: str, y_label_left: str, color_collection = None
        ) -> None:
            '''
                柱状图抽象类型
            '''
            df = self.index_type_judge(df=df)
            colors = self.color_collection_judge(df=df, color_collection=color_collection)
            # df.plot(
            #     kind='bar', ax=ax,
            #     color=colors,
            #     alpha=0.8
            # )
            ax.bar(list(df.index), list(df[df.columns[0]].values), color=colors, alpha=0.8)
            plt.axis()
            ax.legend(labels=list(df.columns), fontsize=self.fig_width // 4, loc='upper left')
            x_labels = list(df.index)
            x_labels_pos = range(0, len(x_labels), self.interval)
            ax.set_xticks(x_labels_pos)
            x_labels = [x for x in x_labels if x_labels.index(x) in x_labels_pos]
            ax.set_xticklabels(x_labels, fontsize = self.fig_width // 5.5)#, rotation=10, ha='right')
            # 设置y轴tick与label
            plt.yticks(fontsize = self.fig_width // 5)
            ax.set_ylabel(y_label_left, fontsize=self.fig_width // 5)
            plt.title(title, fontsize=self.fig_width // 5)
            plt.grid()

        def area_chart(
                self, df: pd.DataFrame, ax: object, title: str, color_collection=None
        ) -> None:
            '''
                面积图抽象类型
            '''
            df = self.index_type_judge(df=df)
            colors = self.color_collection_judge(df=df, color_collection=color_collection)
            df.plot(kind='area', ax=ax, color=colors, alpha=0.5)
            ax.legend(fontsize=self.fig_width // 4, loc='upper left')
            # 设置x轴tick与label
            x_labels = list(df.index)
            x_labels_pos = range(0, len(x_labels), self.interval)
            ax.set_xticks(x_labels_pos)
            x_labels = [x for x in x_labels if x_labels.index(x) in x_labels_pos]
            ax.set_xticklabels(x_labels, fontsize=self.fig_width // 5.5)#, rotation=10, ha='right')
            # 设置y轴tick与label
            plt.yticks(fontsize=self.fig_width // 4.5)
            plt.title(title, fontsize=self.fig_width // 5)
            plt.grid()

        def sheet_chart(self, df: pd.DataFrame, ax: object, title: str) -> None:
            '''
                表格抽象类型
            '''
            tb = ax.table(
                cellText = df.apply(lambda x: round(x, 3)).values,
                colLabels = df.columns,
                rowLabels = df.index,
                loc = 'center',
                cellLoc = 'center',
                rowLoc = 'center',
                edges = 'closed'
            )
            tb.set_fontsize(self.fig_width // 5)
            for _, cell in tb.get_celld().items():
                cell.set_height(self.fig_height / (12 * len(df.index) + 12))
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            # plt.axis('off')
            plt.title(title, fontsize=self.fig_width // 5)


    def cross_day_plot(
        self,
        stat_by_year: dict,
        style_rtn_ts_dict: dict,
        style_exposure_ts_dict: dict,
        industry_exposure_dict: dict,
        style_exposure_dict: dict,
        max_trd_ratio_dict: dict,
        tov_on_index_dict: dict,
        val_SHSZ_ratio_dict: dict,
        vol_SHSZ_ratio_dict: dict,
        val_ratio_dict: dict,
        vol_ratio_dict: dict,
        rtn_dict: dict,
        drawdown_index_dict: dict,
        drawdown_pool_dict: dict,
        save_path = 'test.jpg'
    ) -> None:

        '''
            :param - style_rtn_ts_dict 风格收益时序
            :param - style_exposure_ts_dict 风格暴露时序
            :param - industry_exposure_dict 行业暴露
            :param - style_exposure_dict 风格暴露
            :param - max_trd_ratio_dict 交易金额最大占比三个值
            :param - tov_on_index_dict 指数上对应换手
            :param - val_SHSZ_ratio_dict 持股金额在SH、SZ分布
            :param - vol_SHSZ_ratio_dict 持股数在SH、SZ分布
            :param - val_ratio_dict 持股金额在指数上分布
            :param - vol_ratio_dict 持股数在指数上分布
            :param - rtn_sep_dict 收益拆解
            :param - rtn_dict 收益时序
            :param - fig_width 图像宽度
            :param - fig_height 图像高度
            :param - save_path 存图路径
        '''
        # 生成画布和轴
        figure, axes = plt.subplots(nrows=11, ncols=2, figsize=(self.fig_width, self.fig_height), dpi=240)
        chart = self.Chart(
            fig_width = self.fig_width,
            fig_height = self.fig_height,
            sample_interval = self.sample_interval
        )
        # 左图零，分年度统计表
        ax0 = axes[0, 0]
        ax0 = plt.subplot2grid((11, 10), (0, 0), rowspan=1, colspan=5)
        ax0_df = stat_by_year['df']
        ax0_title = stat_by_year['title']
        chart.sheet_chart(
            df=ax0_df, title=ax0_title, ax=ax0
        )

        # 左图一，收益曲线图
        ax1 = axes[1, 0]
        ax1_color_list = ['red', 'steelblue', 'red', 'steelblue']
        ax1_style_list = ['--', '--', '-', '-']
        ax1 = plt.subplot2grid((11, 10), (1,0), rowspan=2, colspan=5)
        ax1_df = rtn_dict['df']
        ax1_title = rtn_dict['title']
        chart.line_chart(
            df = ax1_df, ax = ax1, title=ax1_title, color_collection=ax1_color_list,
            style_collection=ax1_style_list
        )

        # 左图二，收益拆解图
        # ax2 = axes[2, 0]
        # ax2_color_list = ['steelblue', 'red']
        # ax2 = plt.subplot2grid((11, 10), (3,0), rowspan=2, colspan=5)
        # ax2_df = rtn_sep_dict['df']
        # ax2_title = rtn_sep_dict['title']
        # chart.line_chart(
        #     df = ax2_df, ax = ax2, title=ax2_title, color_collection=ax2_color_list
        # )

        # 左图二，回撤
        ax21_color_list = ['green']
        ax21 = plt.subplot2grid((11, 10), (3, 0), rowspan=1, colspan=5)
        ax21_df = drawdown_index_dict['df']
        ax21_title = drawdown_index_dict['title']
        chart.bar_ts_chart(
            df=ax21_df, ax=ax21, title=ax21_title, color_collection=ax21_color_list,
            y_label_left='Drawdown'
        )

        ax22_color_list = ['green']
        ax22 = plt.subplot2grid((11, 10), (4, 0), rowspan=1, colspan=5)
        ax22_df = drawdown_pool_dict['df']
        ax22_title = drawdown_pool_dict['title']
        chart.bar_ts_chart(
            df=ax22_df, ax=ax22, title=ax22_title, color_collection=ax22_color_list,
            y_label_left='Drawdown'
        )

        # 左图三，持股数占比图
        ax3 = axes[3, 0]
        ax3_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax3 = plt.subplot2grid((11, 12), (5,0), rowspan=2, colspan=3)
        ax3_df = vol_ratio_dict['df']
        ax3_title = vol_ratio_dict['title']
        chart.area_chart(
            df = ax3_df, ax = ax3, title=ax3_title, color_collection=ax3_color_list
        )

        # 左图四，持股金额占比图
        ax4 = axes[4, 0]
        ax4_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax4 = plt.subplot2grid((11, 12), (5,3), rowspan=2, colspan=3)
        ax4_df = val_ratio_dict['df']
        ax4_title = val_ratio_dict['title']
        chart.area_chart(
            df = ax4_df, ax = ax4, title=ax4_title, color_collection=ax4_color_list
        )

        # 左图五，持股数在SH/SZ的占比
        ax5 = axes[5, 0]
        ax5_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax5 = plt.subplot2grid((11, 12), (7,0), rowspan=2, colspan=3)
        ax5_df = vol_SHSZ_ratio_dict['df']
        ax5_title = vol_SHSZ_ratio_dict['title']
        chart.area_chart(
            df = ax5_df, ax = ax5, title=ax5_title, color_collection=ax5_color_list
        )

        # 左图六，持股金额在SH/SZ的占比
        ax6 = axes[6, 0]
        ax6_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax6 = plt.subplot2grid((11, 12), (7,3), rowspan=2, colspan=3)
        ax6_df = val_SHSZ_ratio_dict['df']
        ax6_title = val_SHSZ_ratio_dict['title']
        chart.area_chart(
            df = ax6_df, ax = ax6, title=ax6_title, color_collection=ax6_color_list
        )

        # 左图七，策略在指数上的换手率
        ax7 = axes[7, 0]
        ax7_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax7 = plt.subplot2grid((11, 12), (9,0), rowspan=2, colspan=3)
        ax7_df = tov_on_index_dict['df']
        ax7_title = tov_on_index_dict['title']
        chart.area_chart(
            df = ax7_df, ax = ax7, title=ax7_title, color_collection=ax7_color_list
        )

        # 左图八，最大交易金额占比
        ax8 = axes[8, 0]
        ax8_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax8 = plt.subplot2grid((11, 12), (9,3), rowspan=2, colspan=3)
        ax8_df = max_trd_ratio_dict['df']
        ax8_title = max_trd_ratio_dict['title']
        chart.area_chart(
            df = ax8_df, ax = ax8, title=ax8_title, 
            color_collection=ax8_color_list
        )

        # 右图一，风格因子暴露
        ax9 = axes[0, 1]
        ax9_color_list = ['#1f77b4', '#ff7f0e']
        ax9 = plt.subplot2grid((11, 10), (0,5), rowspan=3, colspan=5)
        ax9_df = style_exposure_dict['df']
        ax9_title = style_exposure_dict['title']
        chart.bar_chart(
            df = ax9_df, ax = ax9, title=ax9_title,
            y_label_left='Exposure', y_label_right='Attribution',
            color_collection=ax9_color_list
        )

        # 右图二，风格因子收益率时序图
        ax10_color_dict = {
            'momentum': 'tan',
            'book_to_price': 'tab:cyan',
            'leverage': 'darkorange',
            'residual_volatility': 'dodgerblue',
            'liquidity': 'green',
            'size': 'gold',
            'non_linear_size': 'grey',
            'earnings_yield': 'r',
            'beta': 'maroon',
            'growth': 'palegreen'
        }
        ax10 = axes[1, 1]
        ax10 = plt.subplot2grid((11, 10), (3,5), rowspan=3, colspan=5)
        ax10_df = style_rtn_ts_dict['df']
        ax10_title = style_rtn_ts_dict['title']
        chart.line_chart(
            df = ax10_df, ax = ax10, title=ax10_title,
            color_collection = ax10_color_dict
        )
        

        # 右图三，风格因子暴露时序图
        ax11_color_dict = {
            'momentum': 'tan',
            'book_to_price': 'tab:cyan',
            'leverage': 'darkorange',
            'residual_volatility': 'dodgerblue',
            'liquidity': 'green',
            'size': 'gold',
            'non_linear_size': 'grey',
            'earnings_yield': 'r',
            'beta': 'maroon',
            'growth': 'palegreen'
        }
        ax11 = axes[2, 1]
        ax11 = plt.subplot2grid((11, 10), (6,5), rowspan=3, colspan=5)
        ax11_df = style_exposure_ts_dict['df']
        ax11_title = style_exposure_ts_dict['title']
        chart.line_chart(
            df = ax11_df, ax = ax11, title=ax11_title,
            color_collection = ax11_color_dict
        )

        # 右图四，行业因子暴露
        ax12 = axes[3, 1]
        ax12_color_list = ['#1f77b4', '#ff7f0e']
        ax12 = plt.subplot2grid((11, 10), (9,5), rowspan=2, colspan=5)
        ax12_df = industry_exposure_dict['df']
        ax12_title = industry_exposure_dict['title']
        chart.bar_chart(
            df = ax12_df, ax = ax12, title=ax12_title,
            y_label_left='Exposure', y_label_right='Attribution',
            color_collection=ax12_color_list
        )

        # 存图
        plt.subplots_adjust(top=self.fig_height / 5)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=1.0)
        # plt.show()

    
    def intra_day_plot(
        self,
        style_rtn_ts_dict: dict,
        style_exposure_ts_dict: dict,
        industry_exposure_dict: dict,
        style_exposure_dict: dict,
        max_trd_ratio_dict: dict,
        tov_on_index_dict: dict,
        val_SHSZ_ratio_dict: dict,
        vol_SHSZ_ratio_dict: dict,
        val_ratio_dict: dict,
        vol_ratio_dict: dict,
        rtn_sep_dict: dict,
        rtn_dict: dict,
        save_path = 'test.jpg'
    ) -> None:

        '''
            :param - style_rtn_ts_dict 风格收益时序
            :param - style_exposure_ts_dict 风格暴露时序
            :param - industry_exposure_dict 行业暴露
            :param - style_exposure_dict 风格暴露
            :param - max_trd_ratio_dict 交易金额最大占比三个值
            :param - tov_on_index_dict 指数上对应换手
            :param - val_SHSZ_ratio_dict 持股金额在SH、SZ分布
            :param - vol_SHSZ_ratio_dict 持股数在SH、SZ分布
            :param - val_ratio_dict 持股金额在指数上分布
            :param - vol_ratio_dict 持股数在指数上分布
            :param - rtn_sep_dict 收益拆解
            :param - rtn_dict 收益时序
            :param - fig_width 图像宽度
            :param - fig_height 图像高度
            :param - save_path 存图路径
        '''
        # 生成画布和轴
        figure, axes = plt.subplots(nrows=8, ncols=2, figsize=(self.fig_width, self.fig_height), dpi=110)
        chart = self.Chart(
            fig_width = self.fig_width,
            fig_height = self.fig_height,
            sample_interval = self.sample_interval
        )

        # 左图一，收益曲线图
        ax1 = axes[0, 0]
        ax1_color_list = ['red', 'steelblue', 'red', 'steelblue']
        ax1_style_list = ['--', '--', '-', '-']
        ax1 = plt.subplot2grid((10, 10), (0,0), rowspan=2, colspan=5)
        ax1_df = rtn_dict['df']
        ax1_title = rtn_dict['title']
        chart.line_chart(
            df = ax1_df, ax = ax1, title=ax1_title, color_collection=ax1_color_list,
            style_collection=ax1_style_list
        )

        # 左图二，收益拆解图
        ax2 = axes[1, 0]
        ax2_color_list = ['steelblue', 'red']
        ax2 = plt.subplot2grid((10, 10), (2,0), rowspan=2, colspan=5)
        ax2_df = rtn_sep_dict['df']
        ax2_title = rtn_sep_dict['title']
        chart.line_chart(
            df = ax2_df, ax = ax2, title=ax2_title, color_collection=ax2_color_list
        )

        # 左图三，持股数占比图
        ax3 = axes[2, 0]
        ax3_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax3 = plt.subplot2grid((10, 12), (4,0), rowspan=2, colspan=3)
        ax3_df = vol_ratio_dict['df']
        ax3_title = vol_ratio_dict['title']
        chart.area_chart(
            df = ax3_df, ax = ax3, title=ax3_title, color_collection=ax3_color_list
        )

        # 左图四，持股金额占比图
        ax4 = axes[3, 0]
        ax4_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax4 = plt.subplot2grid((10, 12), (4, 3), rowspan=2, colspan=3)
        ax4_df = val_ratio_dict['df']
        ax4_title = val_ratio_dict['title']
        chart.area_chart(
            df = ax4_df, ax = ax4, title=ax4_title, color_collection=ax4_color_list
        )

        # 左图五，持股数在SH/SZ的占比
        ax5 = axes[4, 0]
        ax5_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax5 = plt.subplot2grid((10, 12), (6,0), rowspan=2, colspan=3)
        ax5_df = vol_SHSZ_ratio_dict['df']
        ax5_title = vol_SHSZ_ratio_dict['title']
        chart.area_chart(
            df = ax5_df, ax = ax5, title=ax5_title, color_collection=ax5_color_list
        )

        # 左图六，持股金额在SH/SZ的占比
        ax6 = axes[5, 0]
        ax6_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax6 = plt.subplot2grid((10, 12), (6,3), rowspan=2, colspan=3)
        ax6_df = val_SHSZ_ratio_dict['df']
        ax6_title = val_SHSZ_ratio_dict['title']
        chart.area_chart(
            df = ax6_df, ax = ax6, title=ax6_title, color_collection=ax6_color_list
        )

        # 左图七，策略在指数上的换手率
        ax7 = axes[6, 0]
        ax7_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax7 = plt.subplot2grid((10, 12), (8,0), rowspan=2, colspan=3)
        ax7_df = tov_on_index_dict['df']
        ax7_title = tov_on_index_dict['title']
        chart.area_chart(
            df = ax7_df, ax = ax7, title=ax7_title, color_collection=ax7_color_list
        )

        # 左图八，最大交易金额占比
        ax8 = axes[7, 0]
        ax8_color_list = ['chocolate', 'dodgerblue', 'palegreen', 'darkorange', 'grey', 'maroon', 'aquamarine']
        ax8 = plt.subplot2grid((10, 12), (8,3), rowspan=2, colspan=3)
        ax8_df = max_trd_ratio_dict['df']
        ax8_title = max_trd_ratio_dict['title']
        chart.area_chart(
            df = ax8_df, ax = ax8, title=ax8_title, 
            color_collection=ax8_color_list
        )

        # 右图一，风格因子暴露
        ax9 = axes[0, 1]
        ax9 = plt.subplot2grid((10, 10), (0,5), rowspan=2, colspan=5)
        ax9_df = style_exposure_dict['df']
        ax9_title = style_exposure_dict['title']
        chart.bar_chart(
            df = ax9_df, ax = ax9, title=ax9_title,
            y_label_left='Exposure', y_label_right='Attribution'
        )

        # 右图二，风格因子收益率时序图
        ax10_color_dict = {
            'momentum': 'tan',
            'book_to_price': 'tab:cyan',
            'leverage': 'darkorange',
            'residual_volatility': 'dodgerblue',
            'liquidity': 'green',
            'size': 'darkorange',
            'non_linear_size': 'grey',
            'earnings_yield': 'r',
            'beta': 'maroon',
            'growth': 'palegreen'
        }
        ax10 = axes[1, 1]
        ax10 = plt.subplot2grid((10, 10), (2,5), rowspan=3, colspan=5)
        ax10_df = style_rtn_ts_dict['df']
        ax10_title = style_rtn_ts_dict['title']
        chart.line_chart(
            df = ax10_df, ax = ax10, title=ax10_title,
            color_collection = ax10_color_dict
        )
        

        # 右图三，风格因子暴露时序图
        ax11_color_dict = {
            'momentum': 'tan',
            'book_to_price': 'tab:cyan',
            'leverage': 'darkorange',
            'residual_volatility': 'dodgerblue',
            'liquidity': 'green',
            'size': 'darkorange',
            'non_linear_size': 'grey',
            'earnings_yield': 'r',
            'beta': 'maroon',
            'growth': 'palegreen'
        }
        ax11 = axes[2, 1]
        ax11 = plt.subplot2grid((10, 10), (5,5), rowspan=3, colspan=5)
        ax11_df = style_exposure_ts_dict['df']
        ax11_title = style_exposure_ts_dict['title']
        chart.line_chart(
            df = ax11_df, ax = ax11, title=ax11_title,
            color_collection = ax11_color_dict
        )

        # 右图四，行业因子暴露
        ax12 = axes[3, 1]
        ax12 = plt.subplot2grid((10, 10), (8,5), rowspan=2, colspan=5)
        ax12_df = industry_exposure_dict['df']
        ax12_title = industry_exposure_dict['title']
        chart.bar_chart(
            df = ax12_df, ax = ax12, title=ax12_title,
            y_label_left='Exposure', y_label_right='Attribution'
        )

        # 存图
        plt.subplots_adjust(top=self.fig_height / 2.3)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=1.0)
        # plt.show()
